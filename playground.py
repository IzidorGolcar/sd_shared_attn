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
    return float(np.linspace(0.80, 0.30, num_steps)[step])


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
    b_size, seq_len, channels = hidden_states.shape
    _, _, h_full, w_full = fg_mask_2d.shape
    aspect = w_full / float(h_full)
    h_feat = max(1, int(round((seq_len / aspect) ** 0.5)))
    w_feat = max(1, seq_len // h_feat)

    mask_flat = (
        F.interpolate(fg_mask_2d.float(), size=(h_feat, w_feat), mode='nearest')
        .flatten(1) > 0.5
    )

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


def _isolated_self_attention(
    attn,
    hidden_states: torch.Tensor,
    tgt_mask_2d: Optional[torch.Tensor],
    bg_suppress_bias: float = -10000.0,
    temb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    '''
    Runs self-attention while suppressing attention from foreground queries (inside M)
    to background keys (outside M) so surrounding skin textures do not bleed inward.
    '''
    residual = hidden_states
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim
    if input_ndim == 4:
        b, c, h, w = hidden_states.shape
        hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

    batch_size, seq_len, _ = hidden_states.shape
    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    q = attn.to_q(hidden_states)
    k = attn.to_k(hidden_states)
    v = attn.to_v(hidden_states)

    inner_dim = k.shape[-1]
    head_dim = inner_dim // attn.heads

    q = q.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        q = attn.norm_q(q)
    if attn.norm_k is not None:
        k = attn.norm_k(k)

    attn_bias = None
    if tgt_mask_2d is not None and bg_suppress_bias < 0.0:
        _, _, h_full, w_full = tgt_mask_2d.shape
        aspect = w_full / float(h_full)
        h_feat = max(1, int(round((seq_len / aspect) ** 0.5)))
        w_feat = max(1, seq_len // h_feat)

        tgt_down = F.interpolate(tgt_mask_2d, size=(h_feat, w_feat), mode='bilinear', align_corners=False)
        fg_flat = (tgt_down > 0.5).flatten()
        if fg_flat.any() and (~fg_flat).any():
            # Block foreground queries (row=True) from attending to background keys (col=False)
            suppress = fg_flat[:, None] & (~fg_flat[None, :])
            attn_bias = torch.where(
                suppress,
                torch.full((seq_len, seq_len), bg_suppress_bias, device=q.device, dtype=q.dtype),
                torch.zeros((seq_len, seq_len), device=q.device, dtype=q.dtype),
            )[None, None]  # [1, 1, seq_len, seq_len]

    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_bias, dropout_p=0.0, is_causal=False
    )
    out = out.transpose(1, 2).reshape(batch_size, -1, inner_dim).to(q.dtype)

    out = attn.to_out[0](out)
    out = attn.to_out[1](out)

    if input_ndim == 4:
        out = out.transpose(-1, -2).reshape(batch_size, c, h, w)
    if attn.residual_connection:
        out = out + residual

    return out / attn.rescale_output_factor


class BackgroundSuppressedAttnProcessor(AttnProcessor2_0):
    '''
    Applied to non-MasaCtrl self-attention layers (down_blocks, mid_block) to stop
    foreground queries from pulling background facial texture across the mask boundary.
    '''
    def __init__(self, state_dict: dict):
        super().__init__()
        self.state_dict = state_dict

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
        if self.state_dict['mode'] in ('noop', 'read'):
            return _isolated_self_attention(
                attn,
                hidden_states,
                self.state_dict.get('tgt_fg_tensor'),
                bg_suppress_bias=self.state_dict.get('bg_suppress_bias', -10000.0),
                temb=temb,
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


class MultiViewMasaCtrlAttnProcessor(AttnProcessor2_0):
    '''
    MasaCtrl attention processor with foreground-isolated fallback self-attention
    so neither Phase 1 nor the unconditional CFG branch leaks background pores into M.
    '''
    def __init__(self, layer_name: str, state_dict: dict):
        super().__init__()
        self.layer_name = layer_name
        self.state_dict = state_dict

    def _is_layer_active(self) -> bool:
        active_prefixes = self.state_dict.get('active_blocks', ())
        return self.layer_name.startswith(active_prefixes)

    def _get_tgt_blend(self, seq_len: int, dtype: torch.dtype) -> Optional[torch.Tensor]:
        tgt_mask_2d = self.state_dict.get('tgt_fg_tensor')
        if tgt_mask_2d is None:
            return None

        _, _, h_full, w_full = tgt_mask_2d.shape
        aspect = w_full / float(h_full)
        h_feat = max(1, int(round((seq_len / aspect) ** 0.5)))
        w_feat = max(1, seq_len // h_feat)

        tgt_down = F.interpolate(tgt_mask_2d, size=(h_feat, w_feat), mode='bilinear', align_corners=False)
        return (tgt_down > 0.5).to(dtype=dtype).flatten(2).transpose(1, 2)

    def _select_best_view(
        self,
        q_h: torch.Tensor,
        k_bank_h: torch.Tensor,
        valid_mask: torch.Tensor,
        tgt_fg_flat: torch.Tensor,
        record_scores: bool = False,
    ) -> int:
        n_views = k_bank_h.shape[0]
        if n_views == 1 or not tgt_fg_flat.any():
            return 0

        q_fg = F.normalize(q_h[:, :, tgt_fg_flat, :].float(), dim=-1)
        scores = torch.full((n_views,), -1e9, device=q_h.device, dtype=torch.float32)

        for i in range(n_views):
            v_cols = valid_mask[i]
            if not v_cols.any():
                continue
            k_i = F.normalize(k_bank_h[i:i + 1, :, v_cols, :].float(), dim=-1)
            sim = torch.matmul(q_fg, k_i.transpose(-1, -2))
            scores[i] = sim.max(dim=-1).values.mean()

        if record_scores:
            self.state_dict['view_scores'].append(scores.detach())

        return int(torch.argmax(scores).item())

    def _masked_mutual_branch(
        self,
        attn,
        hidden_states_1d: torch.Tensor,
        ref_bank: torch.Tensor,
        valid_mask: torch.Tensor,
        tgt_blend: torch.Tensor,
        record_scores: bool = False,
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
        best_idx = self._select_best_view(
            q_h, k_bank_h, valid_mask, tgt_fg_flat, record_scores=record_scores
        )
        v_cols = valid_mask[best_idx]

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
            if self._is_layer_active():
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

        if mode == 'read' and self._is_layer_active() and self.layer_name in self.state_dict['bank']:
            ref_bank, valid_mask = self.state_dict['bank'][self.layer_name]
            ref_bank = ref_bank.to(device=hidden_states.device, dtype=hidden_states.dtype)
            valid_mask = valid_mask.to(device=hidden_states.device)
            tgt_blend = self._get_tgt_blend(hidden_states.shape[1], hidden_states.dtype)

            if tgt_blend is not None and hidden_states.shape[0] == 2:
                record_scores = bool(self.state_dict.get('record_view_scores', False))
                out_cond = self._masked_mutual_branch(
                    attn,
                    hidden_states[1:],
                    ref_bank,
                    valid_mask,
                    tgt_blend,
                    record_scores=record_scores,
                    temb=temb,
                )
                style_fidelity = self.state_dict.get('style_fidelity', 0.5)
                if style_fidelity > 0.0:
                    out_uncond_ref = self._masked_mutual_branch(
                        attn,
                        hidden_states[:1],
                        ref_bank,
                        valid_mask,
                        tgt_blend,
                        record_scores=False,
                        temb=temb,
                    )
                    out_uncond_self = _isolated_self_attention(
                        attn,
                        hidden_states[:1],
                        self.state_dict.get('tgt_fg_tensor'),
                        bg_suppress_bias=self.state_dict.get('bg_suppress_bias', -10000.0),
                        temb=temb,
                    )
                    out_uncond = style_fidelity * out_uncond_ref + (1.0 - style_fidelity) * out_uncond_self
                else:
                    out_uncond = _isolated_self_attention(
                        attn,
                        hidden_states[:1],
                        self.state_dict.get('tgt_fg_tensor'),
                        bg_suppress_bias=self.state_dict.get('bg_suppress_bias', -10000.0),
                        temb=temb,
                    )
                return torch.cat([out_uncond, out_cond], dim=0)

        # Phase 1 ('noop') or inactive decoder blocks: suppress background -> foreground attention
        return _isolated_self_attention(
            attn,
            hidden_states,
            self.state_dict.get('tgt_fg_tensor'),
            bg_suppress_bias=self.state_dict.get('bg_suppress_bias', -10000.0),
            temb=temb,
        )


class StructuralMultiViewMasaCtrlInpainter:
    def __init__(
        self,
        inpaint_model_id: str = 'sd-legacy/stable-diffusion-inpainting',
        device: str = 'cuda',
        structure_blocks: Tuple[str, ...] = ('up_blocks.1', 'up_blocks.2', 'up_blocks.3'),
        texture_blocks: Tuple[str, ...] = ('up_blocks.2', 'up_blocks.3'),
        batch_chunk_size: int = 6,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        self.structure_blocks = structure_blocks
        self.texture_blocks = texture_blocks
        self.batch_chunk_size = batch_chunk_size

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            inpaint_model_id,
            torch_dtype=self.dtype,
            use_safetensors=False,
            safety_checker=None,
        ).to(self.device)

        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.inv_scheduler = DDIMInverseScheduler.from_config(self.pipe.scheduler.config)

        all_hook_blocks = tuple(dict.fromkeys(structure_blocks + texture_blocks))
        self.state = {
            'mode': 'noop',
            'active_blocks': structure_blocks,
            'bank': {},
            'bank_chunks': {},
            'write_fg_masks': None,
            'tgt_fg_tensor': None,
            'bg_suppress_bias': -10000.0,
            'style_fidelity': 0.5,
            'record_view_scores': False,
            'view_scores': [],
        }
        self._register_attention_processors(all_hook_blocks)

    def _register_attention_processors(self, hook_blocks: Tuple[str, ...]) -> None:
        attn_procs = {}
        for name in self.pipe.unet.attn_processors.keys():
            if name.endswith('attn1.processor'):
                if name.startswith(hook_blocks):
                    attn_procs[name] = MultiViewMasaCtrlAttnProcessor(name, self.state)
                else:
                    attn_procs[name] = BackgroundSuppressedAttnProcessor(self.state)
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
    def _invert_single_reference_null_prompt(
        self,
        ref_latent: torch.Tensor,
        ref_unet_mask: torch.Tensor,
        null_embed: torch.Tensor,
        num_inference_steps: int,
    ) -> Dict[int, torch.Tensor]:
        prev_mode = self.state['mode']
        self.state['mode'] = 'noop'

        self.inv_scheduler.set_timesteps(num_inference_steps, device=self.device)
        latents_f32 = ref_latent.float().clone()
        masked_context = ref_latent * (1.0 - ref_unet_mask)
        trajectory: Dict[int, torch.Tensor] = {}

        for t in self.inv_scheduler.timesteps:
            t_int = int(t.item())
            trajectory[t_int] = latents_f32.to(dtype=self.dtype)

            latent_model_input = self.inv_scheduler.scale_model_input(latents_f32, t).to(dtype=self.dtype)
            unet_input = torch.cat([latent_model_input, ref_unet_mask, masked_context], dim=1)

            noise_pred = self.pipe.unet(
                unet_input,
                t,
                encoder_hidden_states=null_embed,
                return_dict=False,
            )[0]

            latents_f32 = self.inv_scheduler.step(
                noise_pred.float(), t, latents_f32, return_dict=False
            )[0]

        self.state['mode'] = prev_mode
        return trajectory

    @torch.no_grad()
    def generate(
        self,
        init_image: Image.Image,
        inpaint_mask: Image.Image,
        reference_image: Image.Image,
        reference_mask: Image.Image,
        attn_mask: Optional[Image.Image] = None,
        bg_suppress_bias: float = -10000.0,
        prompt: str = 'extreme macro closeup photograph of a single human finger with fingernail and cuticle, shadows',
        negative_prompt: str = '',
        width: int = 512,
        height: int = 512,
        angles_deg: Tuple[float, ...] = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0),
        scales: Tuple[float, ...] = (0.80, 1.00, 1.25),
        num_inference_steps: int = 35,
        switch_step: int = 5,
        structure_end_step: int = 16,
        guidance_scale: float = 6.5,
        style_fidelity: Union[float, Callable[[int, int], float]] = _default_style_fidelity,
        seed: int = 42,
    ) -> Image.Image:
        self.state['bank'].clear()
        self.state['bank_chunks'].clear()
        self.state['view_scores'].clear()
        self.state['record_view_scores'] = False
        self.state['bg_suppress_bias'] = float(bg_suppress_bias)

        # 1. Prepare target image, blending mask, and tight foreground attention mask
        init_img = init_image.convert('RGB').resize((width, height), Image.LANCZOS)
        tgt_mask_arr = np.array(
            inpaint_mask.convert('L').resize((width, height), Image.BILINEAR), dtype=np.float32
        )
        if tgt_mask_arr.max() > 0.0 and tgt_mask_arr.max() < 200.0:
            tgt_mask_arr = np.clip((tgt_mask_arr / tgt_mask_arr.max()) * 255.0, 0.0, 255.0)
        tgt_mask_img = Image.fromarray(tgt_mask_arr.astype(np.uint8), mode='L')

        core_mask_src = attn_mask if attn_mask is not None else inpaint_mask
        core_arr = np.array(
            core_mask_src.convert('L').resize((width, height), Image.NEAREST), dtype=np.float32
        )
        tgt_fg = (core_arr > 127.0).astype(np.float32)
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
        )[:, None].to(device=self.device, dtype=self.dtype)

        latent_chunks = []
        for start in range(0, n_views, self.batch_chunk_size):
            end = min(start + self.batch_chunk_size, n_views)
            lat = self.pipe.vae.encode(ref_tensors[start:end]).latent_dist.mean
            latent_chunks.append((lat * self.pipe.vae.config.scaling_factor).to(dtype=self.dtype))
        ref_latents = torch.cat(latent_chunks, dim=0)
        ref_unet_masks = F.interpolate(ref_fg_tensors, size=(height // 8, width // 8), mode='nearest')

        noise_gen = torch.Generator(device=self.device).manual_seed(seed)
        base_noise = torch.randn(
            (1, ref_latents.shape[1], ref_latents.shape[2], ref_latents.shape[3]),
            generator=noise_gen,
            device=self.device,
            dtype=self.dtype,
        )

        null_embed, _ = self.pipe.encode_prompt(
            prompt='',
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )

        locked = {
            'ready': False,
            'latent': None,
            'unet_mask': None,
            'masked_context': None,
            'fg_tensor': None,
            'trajectory': None,
        }

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
                self.state['record_view_scores'] = False
                return

            self.state['style_fidelity'] = resolve_style_fidelity(step_idx)
            self.state['active_blocks'] = (
                self.structure_blocks if step_idx < structure_end_step else self.texture_blocks
            )
            self.state['bank_chunks'].clear()
            t = timesteps[step_idx]

            if not locked['ready']:
                self.state['record_view_scores'] = n_views > 1
                self.state['view_scores'].clear()
                self.state['mode'] = 'write'

                for start in range(0, n_views, self.batch_chunk_size):
                    end = min(start + self.batch_chunk_size, n_views)
                    b_chunk = end - start

                    self.state['write_fg_masks'] = ref_fg_tensors[start:end]
                    z_ref_t = self.pipe.scheduler.add_noise(
                        ref_latents[start:end],
                        base_noise.expand(b_chunk, -1, -1, -1),
                        t.expand(b_chunk),
                    )
                    z_scaled = self.pipe.scheduler.scale_model_input(z_ref_t, t)
                    masked_ctx = ref_latents[start:end] * (1.0 - ref_unet_masks[start:end])
                    unet_in = torch.cat([z_scaled, ref_unet_masks[start:end], masked_ctx], dim=1)

                    self.pipe.unet(
                        unet_in,
                        t,
                        encoder_hidden_states=null_embed.expand(b_chunk, -1, -1),
                        return_dict=False,
                    )

                self.state['write_fg_masks'] = None
                self._finalize_step_bank()
                self.state['mode'] = 'read'
            else:
                self.state['record_view_scores'] = False
                nearest_t = min(locked['trajectory'].keys(), key=lambda k: abs(k - int(t.item())))
                z_ref_t = locked['trajectory'][nearest_t]
                z_scaled = self.pipe.scheduler.scale_model_input(z_ref_t, t)
                unet_in = torch.cat([z_scaled, locked['unet_mask'], locked['masked_context']], dim=1)

                self.state['write_fg_masks'] = locked['fg_tensor']
                self.state['mode'] = 'write'
                self.pipe.unet(
                    unet_in,
                    t,
                    encoder_hidden_states=null_embed,
                    return_dict=False,
                )
                self.state['write_fg_masks'] = None
                self._finalize_step_bank()
                self.state['mode'] = 'read'

        arm_step(0)

        def step_callback(pipe_obj, step_index, timestep, callback_kwargs):
            if step_index == switch_step and not locked['ready']:
                if self.state['view_scores']:
                    avg_scores = torch.stack(self.state['view_scores'], dim=0).mean(dim=0)
                    best_idx = int(torch.argmax(avg_scores).item())
                else:
                    best_idx = 0

                best_latent = ref_latents[best_idx:best_idx + 1]
                best_unet_mask = ref_unet_masks[best_idx:best_idx + 1]
                best_fg_tensor = ref_fg_tensors[best_idx:best_idx + 1]

                trajectory = self._invert_single_reference_null_prompt(
                    ref_latent=best_latent,
                    ref_unet_mask=best_unet_mask,
                    null_embed=null_embed,
                    num_inference_steps=num_inference_steps,
                )

                locked['latent'] = best_latent
                locked['unet_mask'] = best_unet_mask
                locked['masked_context'] = best_latent * (1.0 - best_unet_mask)
                locked['fg_tensor'] = best_fg_tensor
                locked['trajectory'] = trajectory
                locked['ready'] = True

                self.state['view_scores'].clear()
                self.state['record_view_scores'] = False

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
            self.state['view_scores'].clear()
            self.state['record_view_scores'] = False
            self.state['write_fg_masks'] = None

        return result


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    inpainter = StructuralMultiViewMasaCtrlInpainter(
        inpaint_model_id='sd-legacy/stable-diffusion-inpainting',
        device='cuda',
        structure_blocks=('up_blocks.1', 'up_blocks.2', 'up_blocks.3'),
        texture_blocks=('up_blocks.2', 'up_blocks.3'),
        batch_chunk_size=6,
    )

    init_img = Image.open('assets/src.jpg')
    raw_msk = Image.open('assets/mask_rect.jpg')
    ref_img = Image.open('references/1.jpeg')
    ref_msk = Image.open('references/1_mask.png')

    # Keep blurred mask for smooth compositing, but pass tight raw_msk to attn_mask
    blended_msk = raw_msk.filter(ImageFilter.MaxFilter(25))
    blended_msk = blended_msk.filter(ImageFilter.GaussianBlur(45))

    output = inpainter.generate(
        init_image=init_img,
        inpaint_mask=blended_msk,
        attn_mask=raw_msk,
        bg_suppress_bias=-10000.0,
        reference_image=ref_img,
        reference_mask=ref_msk,
        num_inference_steps=35,
        switch_step=1,
        structure_end_step=16,
        guidance_scale=6.5,
        seed=2,
    )
    output.save('output_isolated_multiview_sd15.png')

    plt.imshow(output)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
