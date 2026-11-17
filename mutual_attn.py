from typing import Callable, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from diffusers import (
    StableDiffusionInpaintPipeline,
    DDIMScheduler,
    DDIMInverseScheduler,
)
from diffusers.models.attention_processor import AttnProcessor2_0


def _constant_style_fidelity(step: int, num_steps: int) -> float:
    return 0.5


class MasaCtrlInpaintAttnProcessor(AttnProcessor2_0):
    '''
    Implements MasaCtrl (Cao et al., ICCV 2023) Eq. (5)-(8) adapted for inpainting:
      - Inside target inpaint mask M: f_o = Attention(Q_tgt, K_ref, V_ref; M_ref)
      - Outside target inpaint mask (1 - M): f_bg = Attention(Q_tgt, K_tgt, V_tgt)
      - Output: f_o * M + f_bg * (1 - M)
    '''
    def __init__(self, layer_name: str, state_dict: dict):
        super().__init__()
        self.layer_name = layer_name
        self.state_dict = state_dict

    def _get_spatial_masks(
        self, seq_len: int, dtype: torch.dtype
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        ref_mask_2d = self.state_dict.get('ref_fg_tensor')  # [1, 1, H, W]
        tgt_mask_2d = self.state_dict.get('tgt_fg_tensor')  # [1, 1, H, W]
        if ref_mask_2d is None or tgt_mask_2d is None:
            return None, None

        _, _, h_full, w_full = tgt_mask_2d.shape
        aspect = w_full / float(h_full)
        h_feat = max(1, int(round((seq_len / aspect) ** 0.5)))
        w_feat = max(1, seq_len // h_feat)

        # M_s: foreground mask of reference image flattened to [1, 1, 1, seq_len] for SDPA attn_mask
        ref_down = F.interpolate(ref_mask_2d, size=(h_feat, w_feat), mode='nearest').flatten(2)
        ref_attn_bias = torch.where(
            ref_down > 0.5,
            torch.zeros_like(ref_down, dtype=dtype),
            torch.full_like(ref_down, -10000.0, dtype=dtype),
        ).unsqueeze(2)  # [1, 1, 1, seq_len]

        # M: foreground mask of target inpaint region flattened to [1, seq_len, 1]
        tgt_down = F.interpolate(tgt_mask_2d, size=(h_feat, w_feat), mode='bilinear', align_corners=False)
        tgt_blend = (tgt_down > 0.5).to(dtype=dtype).flatten(2).transpose(1, 2)  # [1, seq_len, 1]

        return ref_attn_bias, tgt_blend

    def _masked_mutual_branch(
        self,
        attn,
        hidden_states_1d: torch.Tensor,
        ref_states: torch.Tensor,
        ref_attn_bias: torch.Tensor,
        tgt_blend: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states_1d
        if attn.spatial_norm is not None:
            hidden_states_1d = attn.spatial_norm(hidden_states_1d, temb)

        input_ndim = hidden_states_1d.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states_1d.shape
            hidden_states_1d = hidden_states_1d.view(b, c, h * w).transpose(1, 2)

        batch_size, _, _ = hidden_states_1d.shape

        if attn.group_norm is not None:
            hidden_states_1d = attn.group_norm(hidden_states_1d.transpose(1, 2)).transpose(1, 2)

        # Target Q, K, V
        q_tgt = attn.to_q(hidden_states_1d)
        k_tgt = attn.to_k(hidden_states_1d)
        v_tgt = attn.to_v(hidden_states_1d)

        # Reference K_s, V_s (direct replacement per MasaCtrl Eq. 5-6)
        k_ref = attn.to_k(ref_states)
        v_ref = attn.to_v(ref_states)

        inner_dim = k_tgt.shape[-1]
        head_dim = inner_dim // attn.heads

        q_h = q_tgt.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        k_tgt_h = k_tgt.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        v_tgt_h = v_tgt.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        k_ref_h = k_ref.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        v_ref_h = v_ref.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            q_h = attn.norm_q(q_h)
        if attn.norm_k is not None:
            k_tgt_h = attn.norm_k(k_tgt_h)
            k_ref_h = attn.norm_k(k_ref_h)

        # Eq. (6): Foreground queries attend exclusively to foreground reference keys/values (M_s)
        f_obj = F.scaled_dot_product_attention(
            q_h, k_ref_h, v_ref_h, attn_mask=ref_attn_bias, dropout_p=0.0, is_causal=False
        )
        f_obj = f_obj.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(q_tgt.dtype)

        # Background preserves standard self-attention
        f_bg = F.scaled_dot_product_attention(
            q_h, k_tgt_h, v_tgt_h, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        f_bg = f_bg.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(q_tgt.dtype)

        # Eq. (8): Spatial mask combination f_bar = f_obj * M + f_bg * (1 - M)
        out = f_obj * tgt_blend + f_bg * (1.0 - tgt_blend)

        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        if input_ndim == 4:
            out = out.transpose(-1, -2).reshape(batch_size, c, h, w)

        if attn.residual_connection:
            out = out + residual

        return out / attn.rescale_output_factor

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        mode = self.state_dict['mode']

        if mode == 'write':
            if not torch.isnan(hidden_states).any():
                self.state_dict['bank'][self.layer_name] = hidden_states.detach()
            return super().__call__(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                temb=temb,
                *args,
                **kwargs,
            )

        if mode == 'read' and self.layer_name in self.state_dict['bank']:
            ref_states = self.state_dict['bank'][self.layer_name].to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            ref_attn_bias, tgt_blend = self._get_spatial_masks(
                hidden_states.shape[1], hidden_states.dtype
            )

            if ref_attn_bias is not None and tgt_blend is not None:
                # CFG batch size = 2: [uncond (0), cond (1)]
                if hidden_states.shape[0] == 2 * ref_states.shape[0]:
                    out_cond = self._masked_mutual_branch(
                        attn, hidden_states[1:], ref_states, ref_attn_bias, tgt_blend, temb=temb
                    )
                    style_fidelity = self.state_dict.get('style_fidelity', 0.5)
                    if style_fidelity > 0.0:
                        out_uncond_ref = self._masked_mutual_branch(
                            attn, hidden_states[:1], ref_states, ref_attn_bias, tgt_blend, temb=temb
                        )
                        out_uncond_self = super().__call__(
                            attn,
                            hidden_states[:1],
                            encoder_hidden_states=None,
                            attention_mask=attention_mask,
                            temb=temb,
                            *args,
                            **kwargs,
                        )
                        out_uncond = style_fidelity * out_uncond_ref + (1.0 - style_fidelity) * out_uncond_self
                    else:
                        out_uncond = super().__call__(
                            attn,
                            hidden_states[:1],
                            encoder_hidden_states=None,
                            attention_mask=attention_mask,
                            temb=temb,
                            *args,
                            **kwargs,
                        )
                    return torch.cat([out_uncond, out_cond], dim=0)

        return super().__call__(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            *args,
            **kwargs,
        )


class MasaCtrlInpainter:
    '''
    Two-phase zero-shot SD 1.5 Inpainting pipeline:
      Phase 1 (steps 0 .. switch_step - 1): text prompt layout generation.
      Phase 2 (steps switch_step .. N - 1): MasaCtrl mask-guided mutual self-attention
                                            transfers reference region M_s -> M.
    '''
    def __init__(
        self,
        inpaint_model_id: str = 'sd-legacy/stable-diffusion-inpainting',
        device: str = 'cuda',
        hook_blocks: Tuple[str, ...] = ('up_blocks',),  # Decoder layers per MasaCtrl Sec. 4.1 & Fig. 12(b)
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float16 if self.device.type == 'cuda' else torch.float32

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            inpaint_model_id,
            torch_dtype=self.dtype,
            use_safetensors=False,
            safety_checker=None,
        ).to(self.device)

        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.inv_scheduler = DDIMInverseScheduler.from_config(self.pipe.scheduler.config)

        self.state = {
            'mode': 'noop',
            'bank': {},
            'ref_fg_tensor': None,
            'tgt_fg_tensor': None,
            'style_fidelity': 0.5,
        }
        self._register_attention_processors(hook_blocks)

    def _register_attention_processors(self, hook_blocks: Tuple[str, ...]) -> None:
        attn_procs = {}
        for name in self.pipe.unet.attn_processors.keys():
            if name.startswith(hook_blocks) and name.endswith('attn1.processor'):
                attn_procs[name] = MasaCtrlInpaintAttnProcessor(name, self.state)
            else:
                attn_procs[name] = AttnProcessor2_0()
        self.pipe.unet.set_attn_processor(attn_procs)

    @torch.no_grad()
    def _invert_reference_null_prompt(
        self,
        ref_latents: torch.Tensor,
        ref_unet_mask: torch.Tensor,
        num_inference_steps: int,
    ) -> Dict[int, torch.Tensor]:
        '''
        Deterministic DDIM inversion with null prompt P_s = '' (MasaCtrl Footnote 2).
        Scheduler arithmetic runs in float32 to prevent float16 overflow near t -> T.
        '''
        self.state['mode'] = 'noop'

        prompt_embeds, _ = self.pipe.encode_prompt(
            prompt='',
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )

        self.inv_scheduler.set_timesteps(num_inference_steps, device=self.device)
        latents_f32 = ref_latents.float().clone()
        trajectory: Dict[int, torch.Tensor] = {}

        for t in self.inv_scheduler.timesteps:
            t_int = int(t.item())
            trajectory[t_int] = latents_f32.to(dtype=self.dtype)

            latent_model_input = self.inv_scheduler.scale_model_input(latents_f32, t).to(dtype=self.dtype)
            unet_input = torch.cat([latent_model_input, ref_unet_mask, ref_latents], dim=1)

            noise_pred = self.pipe.unet(
                unet_input,
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

            latents_f32 = self.inv_scheduler.step(
                noise_pred.float(), t, latents_f32, return_dict=False
            )[0]

        return trajectory

    @torch.no_grad()
    def generate(
        self,
        init_image: Image.Image,
        inpaint_mask: Image.Image,
        reference_image: Image.Image,
        reference_mask: Image.Image,
        prompt: str = 'extreme macro closeup photograph of a single human finger, shadows',
        negative_prompt: str = '',
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 30,
        switch_step: int = 6,
        guidance_scale: float = 6.5,
        style_fidelity: Callable[[int, int], float] = _constant_style_fidelity,
        crop_to_ref_mask: bool = True,
        seed: int = 42,
    ) -> Image.Image:
        self.state['bank'].clear()
        self.state['style_fidelity'] = 0.5

        # 1. Align inputs
        init_img = init_image.convert('RGB').resize((width, height), Image.LANCZOS)
        tgt_mask_img = inpaint_mask.convert('L').resize((width, height), Image.NEAREST)

        ref_rgb = reference_image.convert('RGB')
        ref_mask_l = reference_mask.convert('L').resize(ref_rgb.size, Image.NEAREST)

        if crop_to_ref_mask:
            mask_arr = np.array(ref_mask_l)
            ys, xs = np.where(mask_arr > 127)
            if len(xs) > 0 and len(ys) > 0:
                h_orig, w_orig = mask_arr.shape
                pad_x = int((xs.max() - xs.min()) * 0.08)
                pad_y = int((ys.max() - ys.min()) * 0.08)
                x0 = max(0, xs.min() - pad_x)
                y0 = max(0, ys.min() - pad_y)
                x1 = min(w_orig, xs.max() + pad_x)
                y1 = min(h_orig, ys.max() + pad_y)
                ref_rgb = ref_rgb.crop((x0, y0, x1, y1))
                ref_mask_l = ref_mask_l.crop((x0, y0, x1, y1))

        ref_img = ref_rgb.resize((width, height), Image.LANCZOS)
        ref_mask_img = ref_mask_l.resize((width, height), Image.NEAREST)

        # Store M and M_s in shared state for Eq. (6)-(8)
        tgt_fg = (np.array(tgt_mask_img, dtype=np.uint8) > 127).astype(np.float32)
        ref_fg = (np.array(ref_mask_img, dtype=np.uint8) > 127).astype(np.float32)
        self.state['tgt_fg_tensor'] = torch.from_numpy(tgt_fg)[None, None].to(device=self.device, dtype=self.dtype)
        self.state['ref_fg_tensor'] = torch.from_numpy(ref_fg)[None, None].to(device=self.device, dtype=self.dtype)

        # 2. Encode reference latents and invert with null prompt P_s = ''
        ref_tensor = self.pipe.image_processor.preprocess(ref_img, height=height, width=width).to(
            device=self.device, dtype=self.dtype
        )
        ref_latents = (self.pipe.vae.encode(ref_tensor).latent_dist.mean * self.pipe.vae.config.scaling_factor).to(
            dtype=self.dtype
        )
        ref_unet_mask = F.interpolate(self.state['ref_fg_tensor'], size=(height // 8, width // 8), mode='nearest')

        ref_trajectory = self._invert_reference_null_prompt(
            ref_latents=ref_latents,
            ref_unet_mask=ref_unet_mask,
            num_inference_steps=num_inference_steps,
        )

        # 3. Prepare Phase 2 reference feature extractor
        null_embeds, _ = self.pipe.encode_prompt(
            prompt='',
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )

        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps

        def get_style_fidelity(step_idx: int) -> float:
            fidelity = style_fidelity(step_idx, len(timesteps))
            if not 0.0 <= fidelity <= 1.0:
                raise ValueError('style_fidelity must return a value between 0.0 and 1.0')
            return fidelity

        def arm_step(step_idx: int) -> None:
            # Phase 1: text prompt layout synthesis
            if step_idx < switch_step or step_idx >= len(timesteps):
                self.state['mode'] = 'noop'
                return

            self.state['style_fidelity'] = get_style_fidelity(step_idx)

            # Phase 2: extract K_s, V_s from inverted trajectory and activate read mode
            t = timesteps[step_idx]
            nearest_t = min(ref_trajectory.keys(), key=lambda k: abs(k - int(t.item())))
            z_ref_t = ref_trajectory[nearest_t]
            z_ref_scaled = self.pipe.scheduler.scale_model_input(z_ref_t, t)
            unet_input = torch.cat([z_ref_scaled, ref_unet_mask, ref_latents], dim=1)

            self.state['mode'] = 'write'
            self.pipe.unet(
                unet_input,
                t,
                encoder_hidden_states=null_embeds,
                return_dict=False,
            )
            self.state['mode'] = 'read'

        arm_step(0)

        def step_callback(pipe_obj, step_index, timestep, callback_kwargs):
            arm_step(step_index + 1)
            return callback_kwargs

        try:
            gen = torch.Generator(device=self.device).manual_seed(seed)
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_img,
                mask_image=tgt_mask_img,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=gen,
                callback_on_step_end=step_callback,
            ).images[0]
        finally:
            self.state['mode'] = 'noop'
            self.state['bank'].clear()

        return result


if __name__ == '__main__':
    inpainter = MasaCtrlInpainter(device='cuda')

    init_img = Image.open('assets/src.jpg')
    inpaint_msk = Image.open('assets/mask_rect.jpg')
    ref_img = Image.open('references/1.jpeg')
    ref_msk = Image.open('references/1_mask.png')

    inpaint_msk = inpaint_msk.filter(ImageFilter.MaxFilter(25))
    inpaint_msk = inpaint_msk.filter(ImageFilter.GaussianBlur(45))

    import matplotlib.pyplot as plt


    def fidelity_func(s, n):
        if s < n / 3:
            return 0.0
        # if s < n / 5:
        #     return 0.0
        # return np.random.uniform(0.0, 1.0)
        return np.linspace(0.8, 0.0, n)[s]
        # return np.arange(0.0, 0.7, 0.7 / n)[s]
        # return min([1.0, max([0.0, np.random.normal(loc=0.5, scale=0.4)])])

    output = inpainter.generate(
        init_image=init_img,
        inpaint_mask=inpaint_msk,
        reference_image=ref_img,
        reference_mask=ref_msk,
        num_inference_steps=40,
        switch_step=5,
        seed=1,
        style_fidelity=fidelity_func,
        guidance_scale=6,
        # prompt='',
        crop_to_ref_mask=False
    )
    output.save('output_masactrl_two_phase_sd15.png')
    plt.imshow(output)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
