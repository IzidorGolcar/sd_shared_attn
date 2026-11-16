import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from diffusers import (
    DDIMInverseScheduler,
    DDIMScheduler,
    StableDiffusionInpaintPipeline,
)
from diffusers.models.attention_processor import AttnProcessor2_0


def _default_style_fidelity(step: int, num_steps: int) -> float:
    return 0.5


def _mask_centroid(mask_l: Image.Image) -> Tuple[float, float]:
    arr = np.array(mask_l, dtype=np.uint8) > 127
    ys, xs = np.where(arr)
    if len(xs) == 0:
        h, w = arr.shape
        return w / 2.0, h / 2.0
    return float(xs.mean()), float(ys.mean())


def _build_augmented_views(
    reference_image: Image.Image,
    reference_mask: Image.Image,
    width: int,
    height: int,
    angles_deg: Tuple[float, ...],
    scales: Tuple[float, ...],
) -> Tuple[List[Image.Image], List[Image.Image]]:
    '''
    Precomputes rotation- and scale-augmented reference views centered on the
    foreground mask centroid using reflection padding so edges never clip.
    '''
    ref_rgb = reference_image.convert('RGB')
    ref_msk = reference_mask.convert('L').resize(ref_rgb.size, Image.NEAREST)

    w_orig, h_orig = ref_rgb.size
    cx, cy = _mask_centroid(ref_msk)
    base_scale = min(width / float(w_orig), height / float(h_orig))

    pad = max(w_orig, h_orig)
    ref_pad = Image.fromarray(
        np.pad(np.array(ref_rgb), ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    )
    msk_pad = Image.fromarray(
        np.pad(np.array(ref_msk), ((pad, pad), (pad, pad)), mode='constant', constant_values=0)
    )

    ix, iy = cx + pad, cy + pad
    ox, oy = width / 2.0, height / 2.0

    aug_images: List[Image.Image] = []
    aug_masks: List[Image.Image] = []

    for scale in scales:
        inv_s = 1.0 / (base_scale * scale)
        for angle_deg in angles_deg:
            rad = math.radians(angle_deg)
            cos_a, sin_a = math.cos(rad), math.sin(rad)

            a = inv_s * cos_a
            b = inv_s * sin_a
            c = ix - (a * ox + b * oy)
            d = -inv_s * sin_a
            e = inv_s * cos_a
            f = iy - (d * ox + e * oy)
            coeffs = (a, b, c, d, e, f)

            img_view = ref_pad.transform((width, height), Image.AFFINE, coeffs, resample=Image.BICUBIC)
            msk_view = msk_pad.transform((width, height), Image.AFFINE, coeffs, resample=Image.NEAREST)

            aug_images.append(img_view)
            aug_masks.append(msk_view)

    return aug_images, aug_masks


def compact_foreground_tokens(
    hidden_states: torch.Tensor,  # [B, seq_len, C]
    fg_mask_2d: torch.Tensor,     # [B, 1, H, W]
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''Strips background tokens per view and packs foreground tokens to [B, M_max, C].'''
    b_size, seq_len, channels = hidden_states.shape
    _, _, h_full, w_full = fg_mask_2d.shape
    aspect = w_full / float(h_full)
    h_feat = max(1, int(round((seq_len / aspect) ** 0.5)))
    w_feat = max(1, seq_len // h_feat)

    mask_flat = (
        F.interpolate(fg_mask_2d.float(), size=(h_feat, w_feat), mode='nearest')
        .flatten(1) > 0.5
    )  # [B, seq_len]

    counts = mask_flat.sum(dim=1)
    m_max = max(int(counts.max().item()), 1)

    compacted = torch.zeros(
        (b_size, m_max, channels), device=hidden_states.device, dtype=hidden_states.dtype
    )
    valid_mask = torch.zeros((b_size, m_max), device=hidden_states.device, dtype=torch.bool)

    for i in range(b_size):
        idx = torch.where(mask_flat[i])[0]
        if len(idx) == 0:
            idx = torch.arange(min(seq_len, m_max), device=hidden_states.device)
        compacted[i, :len(idx)] = hidden_states[i, idx]
        valid_mask[i, :len(idx)] = True

    return compacted, valid_mask


class MultiViewMasaCtrlAttnProcessor(AttnProcessor2_0):
    '''
    Multi-view MasaCtrl attention processor with foreground token compaction
    and Best-View Routing to preserve sharp 1:1 reference texture across
    precomputed rotation and scale augmentations.
    '''
    def __init__(self, layer_name: str, state_dict: dict):
        super().__init__()
        self.layer_name = layer_name
        self.state_dict = state_dict

    def _get_tgt_blend(self, seq_len: int, dtype: torch.dtype) -> Optional[torch.Tensor]:
        tgt_mask_2d = self.state_dict.get('tgt_fg_tensor')  # [1, 1, H, W]
        if tgt_mask_2d is None:
            return None

        _, _, h_full, w_full = tgt_mask_2d.shape
        aspect = w_full / float(h_full)
        h_feat = max(1, int(round((seq_len / aspect) ** 0.5)))
        w_feat = max(1, seq_len // h_feat)

        tgt_down = F.interpolate(tgt_mask_2d, size=(h_feat, w_feat), mode='bilinear', align_corners=False)
        return (tgt_down > 0.5).to(dtype=dtype).flatten(2).transpose(1, 2)  # [1, seq_len, 1]

    def _select_best_view(
        self,
        q_h: torch.Tensor,          # [1, heads, L_q, head_dim]
        k_bank_h: torch.Tensor,     # [N, heads, M_max, head_dim]
        valid_mask: torch.Tensor,   # [N, M_max]
        tgt_fg_flat: torch.Tensor,  # [L_q] bool
    ) -> int:
        n_views = k_bank_h.shape[0]
        if n_views == 1 or not tgt_fg_flat.any():
            return 0

        q_fg = F.normalize(q_h[:, :, tgt_fg_flat, :].float(), dim=-1)  # [1, heads, L_fg, head_dim]
        best_idx = 0
        best_score = -1e9

        for i in range(n_views):
            v_cols = valid_mask[i]
            if not v_cols.any():
                continue
            k_i = F.normalize(k_bank_h[i:i + 1, :, v_cols, :].float(), dim=-1)
            sim = torch.matmul(q_fg, k_i.transpose(-1, -2))  # [1, heads, L_fg, M_i]
            score = sim.max(dim=-1).values.mean().item()
            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

    def _masked_mutual_branch(
        self,
        attn,
        hidden_states_1d: torch.Tensor,
        ref_bank: torch.Tensor,     # [N, M_max, C]
        valid_mask: torch.Tensor,   # [N, M_max]
        tgt_blend: torch.Tensor,    # [1, seq_len, 1]
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

        q_tgt = attn.to_q(hidden_states_1d)
        k_tgt = attn.to_k(hidden_states_1d)
        v_tgt = attn.to_v(hidden_states_1d)

        k_bank = attn.to_k(ref_bank)
        v_bank = attn.to_v(ref_bank)

        inner_dim = k_tgt.shape[-1]
        head_dim = inner_dim // attn.heads

        q_h = q_tgt.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        k_tgt_h = k_tgt.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        v_tgt_h = v_tgt.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        k_bank_h = k_bank.view(ref_bank.shape[0], -1, attn.heads, head_dim).transpose(1, 2)
        v_bank_h = v_bank.view(ref_bank.shape[0], -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            q_h = attn.norm_q(q_h)
        if attn.norm_k is not None:
            k_tgt_h = attn.norm_k(k_tgt_h)
            k_bank_h = attn.norm_k(k_bank_h)

        tgt_fg_flat = tgt_blend[0, :, 0] > 0.5
        best_idx = self._select_best_view(q_h, k_bank_h, valid_mask, tgt_fg_flat)
        v_cols = valid_mask[best_idx]

        # Sharp 1:1 attention against only the winning orientation/scale view
        f_obj = F.scaled_dot_product_attention(
            q_h,
            k_bank_h[best_idx:best_idx + 1, :, v_cols, :],
            v_bank_h[best_idx:best_idx + 1, :, v_cols, :],
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        f_obj = f_obj.transpose(1, 2).reshape(batch_size, -1, inner_dim).to(q_tgt.dtype)

        f_bg = F.scaled_dot_product_attention(
            q_h, k_tgt_h, v_tgt_h, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        f_bg = f_bg.transpose(1, 2).reshape(batch_size, -1, inner_dim).to(q_tgt.dtype)

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
            write_masks = self.state_dict.get('write_fg_masks')
            if write_masks is not None and not torch.isnan(hidden_states).any():
                compacted, valid_mask = compact_foreground_tokens(
                    hidden_states.detach(), write_masks
                )
                self.state_dict['bank_chunks'].setdefault(self.layer_name, []).append(
                    (compacted, valid_mask)
                )
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
            ref_bank, valid_mask = self.state_dict['bank'][self.layer_name]
            ref_bank = ref_bank.to(device=hidden_states.device, dtype=hidden_states.dtype)
            valid_mask = valid_mask.to(device=hidden_states.device)
            tgt_blend = self._get_tgt_blend(hidden_states.shape[1], hidden_states.dtype)

            # CFG batch size = 2: [uncond (0), cond (1)]
            if tgt_blend is not None and hidden_states.shape[0] == 2:
                out_cond = self._masked_mutual_branch(
                    attn, hidden_states[1:], ref_bank, valid_mask, tgt_blend, temb=temb
                )
                style_fidelity = self.state_dict.get('style_fidelity', 0.5)
                if style_fidelity > 0.0:
                    out_uncond_ref = self._masked_mutual_branch(
                        attn, hidden_states[:1], ref_bank, valid_mask, tgt_blend, temb=temb
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


class MultiViewMasaCtrlInpainter:
    '''
    Zero-shot SD 1.5 Inpainting pipeline with precomputed multi-angle and
    multi-scale reference feature banks and Best-View Routing.
    '''
    def __init__(
        self,
        inpaint_model_id: str = 'sd-legacy/stable-diffusion-inpainting',
        device: str = 'cuda',
        hook_blocks: Tuple[str, ...] = ('up_blocks.2', 'up_blocks.3'),
        batch_chunk_size: int = 6,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        self.batch_chunk_size = batch_chunk_size

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
            'bank_chunks': {},
            'write_fg_masks': None,
            'tgt_fg_tensor': None,
            'style_fidelity': 0.5,
        }
        self._register_attention_processors(hook_blocks)

    def _register_attention_processors(self, hook_blocks: Tuple[str, ...]) -> None:
        attn_procs = {}
        for name in self.pipe.unet.attn_processors.keys():
            if name.startswith(hook_blocks) and name.endswith('attn1.processor'):
                attn_procs[name] = MultiViewMasaCtrlAttnProcessor(name, self.state)
            else:
                attn_procs[name] = AttnProcessor2_0()
        self.pipe.unet.set_attn_processor(attn_procs)

    def _finalize_step_bank(self) -> None:
        self.state['bank'].clear()
        for layer_name, chunks in self.state['bank_chunks'].items():
            n_total = sum(comp.shape[0] for comp, _ in chunks)
            m_max = max(comp.shape[1] for comp, _ in chunks)
            channels = chunks[0][0].shape[2]

            full_comp = torch.zeros((n_total, m_max, channels), device=self.device, dtype=self.dtype)
            full_mask = torch.zeros((n_total, m_max), device=self.device, dtype=torch.bool)

            offset = 0
            for comp, valid in chunks:
                b_k, m_k, _ = comp.shape
                full_comp[offset:offset + b_k, :m_k] = comp
                full_mask[offset:offset + b_k, :m_k] = valid
                offset += b_k

            self.state['bank'][layer_name] = (full_comp, full_mask)
        self.state['bank_chunks'].clear()

    @torch.no_grad()
    def _invert_references_null_prompt(
        self,
        ref_latents: torch.Tensor,    # [N, 4, H//8, W//8]
        ref_unet_masks: torch.Tensor, # [N, 1, H//8, W//8]
        num_inference_steps: int,
    ) -> Dict[int, torch.Tensor]:
        self.state['mode'] = 'noop'
        n_views = ref_latents.shape[0]

        null_embed, _ = self.pipe.encode_prompt(
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

            noise_preds = []
            for start in range(0, n_views, self.batch_chunk_size):
                end = min(start + self.batch_chunk_size, n_views)
                b_chunk = end - start

                lat_chunk = self.inv_scheduler.scale_model_input(
                    latents_f32[start:end], t
                ).to(dtype=self.dtype)
                unet_in = torch.cat(
                    [lat_chunk, ref_unet_masks[start:end], ref_latents[start:end]], dim=1
                )
                pred_chunk = self.pipe.unet(
                    unet_in,
                    t,
                    encoder_hidden_states=null_embed.expand(b_chunk, -1, -1),
                    return_dict=False,
                )[0]
                noise_preds.append(pred_chunk)

            noise_pred = torch.cat(noise_preds, dim=0)
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
        angles_deg: Tuple[float, ...] = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0),
        scales: Tuple[float, ...] = (0.80, 1.00, 1.25),
        num_inference_steps: int = 35,
        switch_step: int = 10,
        guidance_scale: float = 6.5,
        style_fidelity: Union[float, Callable[[int, int], float]] = _default_style_fidelity,
        seed: int = 42,
    ) -> Image.Image:
        self.state['bank'].clear()
        self.state['bank_chunks'].clear()

        # 1. Prepare target image & mask (normalize peak if heavily blurred)
        init_img = init_image.convert('RGB').resize((width, height), Image.LANCZOS)
        tgt_mask_arr = np.array(
            inpaint_mask.convert('L').resize((width, height), Image.BILINEAR), dtype=np.float32
        )
        if tgt_mask_arr.max() > 0.0 and tgt_mask_arr.max() < 200.0:
            tgt_mask_arr = np.clip((tgt_mask_arr / tgt_mask_arr.max()) * 255.0, 0.0, 255.0)
        tgt_mask_img = Image.fromarray(tgt_mask_arr.astype(np.uint8), mode='L')

        tgt_fg = (tgt_mask_arr > 127.0).astype(np.float32)
        self.state['tgt_fg_tensor'] = torch.from_numpy(tgt_fg)[None, None].to(
            device=self.device, dtype=self.dtype
        )

        # 2. Precompute rotation & scale augmented reference views
        aug_images, aug_masks = _build_augmented_views(
            reference_image=reference_image,
            reference_mask=reference_mask,
            width=width,
            height=height,
            angles_deg=angles_deg,
            scales=scales,
        )
        n_views = len(aug_images)

        ref_tensors = torch.cat(
            [
                self.pipe.image_processor.preprocess(img, height=height, width=width)
                for img in aug_images
            ],
            dim=0,
        ).to(device=self.device, dtype=self.dtype)

        ref_fg_tensors = torch.from_numpy(
            np.stack([(np.array(m, dtype=np.uint8) > 127).astype(np.float32) for m in aug_masks], axis=0)
        )[:, None].to(device=self.device, dtype=self.dtype)  # [N, 1, H, W]

        # Encode all views in chunks to keep peak VRAM low
        latent_chunks = []
        for start in range(0, n_views, self.batch_chunk_size):
            end = min(start + self.batch_chunk_size, n_views)
            lat = self.pipe.vae.encode(ref_tensors[start:end]).latent_dist.mean
            latent_chunks.append((lat * self.pipe.vae.config.scaling_factor).to(dtype=self.dtype))
        ref_latents = torch.cat(latent_chunks, dim=0)  # [N, 4, H//8, W//8]
        ref_unet_masks = F.interpolate(ref_fg_tensors, size=(height // 8, width // 8), mode='nearest')

        # 3. Deterministic DDIM inversion across all N reference views
        ref_trajectory = self._invert_references_null_prompt(
            ref_latents=ref_latents,
            ref_unet_masks=ref_unet_masks,
            num_inference_steps=num_inference_steps,
        )

        null_embed, _ = self.pipe.encode_prompt(
            prompt='',
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )

        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps

        def resolve_style_fidelity(step_idx: int) -> float:
            val = float(style_fidelity(step_idx, len(timesteps))) if callable(style_fidelity) else float(style_fidelity)
            if not 0.0 <= val <= 1.0:
                raise ValueError('style_fidelity must be between 0.0 and 1.0')
            return val

        def arm_step(step_idx: int) -> None:
            if step_idx < switch_step or step_idx >= len(timesteps):
                self.state['mode'] = 'noop'
                return

            self.state['style_fidelity'] = resolve_style_fidelity(step_idx)
            self.state['bank_chunks'].clear()

            t = timesteps[step_idx]
            nearest_t = min(ref_trajectory.keys(), key=lambda k: abs(k - int(t.item())))
            z_ref_t = ref_trajectory[nearest_t]

            self.state['mode'] = 'write'
            for start in range(0, n_views, self.batch_chunk_size):
                end = min(start + self.batch_chunk_size, n_views)
                b_chunk = end - start

                self.state['write_fg_masks'] = ref_fg_tensors[start:end]
                z_scaled = self.pipe.scheduler.scale_model_input(z_ref_t[start:end], t)
                unet_in = torch.cat(
                    [z_scaled, ref_unet_masks[start:end], ref_latents[start:end]], dim=1
                )
                self.pipe.unet(
                    unet_in,
                    t,
                    encoder_hidden_states=null_embed.expand(b_chunk, -1, -1),
                    return_dict=False,
                )

            self.state['write_fg_masks'] = None
            self._finalize_step_bank()
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
            self.state['bank_chunks'].clear()
            self.state['write_fg_masks'] = None

        return result


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    inpainter = MultiViewMasaCtrlInpainter(
        inpaint_model_id='sd-legacy/stable-diffusion-inpainting',
        device='cuda',
        hook_blocks=('up_blocks.2', 'up_blocks.3'),
        batch_chunk_size=6,
    )

    init_img = Image.open('assets/src.jpg')
    inpaint_msk = Image.open('assets/mask_rect.jpg')
    ref_img = Image.open('references/1.jpeg')
    ref_msk = Image.open('references/1_mask.png')

    inpaint_msk = inpaint_msk.filter(ImageFilter.MaxFilter(25))
    inpaint_msk = inpaint_msk.filter(ImageFilter.GaussianBlur(45))

    def fidelity_func(s: int, n: int) -> float:
        return float(np.linspace(0.1, 0.8, n)[s])

    output = inpainter.generate(
        init_image=init_img,
        inpaint_mask=inpaint_msk,
        reference_image=ref_img,
        reference_mask=ref_msk,
        angles_deg=tuple(np.linspace(-30, 30, 8)),
        # (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0),
        scales=tuple(np.linspace(0.80, 1.25, 5)),
        # (0.80, 1.00, 1.25),
        num_inference_steps=35,
        switch_step=5,
        guidance_scale=6.5,
        style_fidelity=fidelity_func,
        seed=42,
    )
    output.save('output_multiview_masactrl_sd15.png')

    plt.imshow(output)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
