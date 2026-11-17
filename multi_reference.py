import math
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from diffusers import (
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


def _build_spatial_isolation_bias(
    mask_2d: torch.Tensor,
    seq_len: int,
    bg_suppress_bias: float,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    '''
    Builds a [B, 1, seq_len, seq_len] attention bias suppressing foreground queries
    (inside mask) from attending to background keys (outside mask).
    '''
    b_mask, _, h_full, w_full = mask_2d.shape
    aspect = w_full / float(h_full)
    h_feat = max(1, int(round((seq_len / aspect) ** 0.5)))
    w_feat = max(1, seq_len // h_feat)

    mask_down = F.interpolate(mask_2d.float(), size=(h_feat, w_feat), mode='bilinear', align_corners=False)
    fg_flat = (mask_down > 0.5).view(b_mask, seq_len)

    # Block foreground queries (row=True) from attending to background keys (col=False)
    suppress = fg_flat[:, :, None] & (~fg_flat[:, None, :])  # [B, seq_len, seq_len]
    if not suppress.any():
        return None

    bias_val = torch.tensor(bg_suppress_bias, device=device, dtype=dtype)
    zero_val = torch.tensor(0.0, device=device, dtype=dtype)
    return torch.where(suppress, bias_val, zero_val)[:, None, :, :]  # [B, 1, seq_len, seq_len]


def _get_isolation_bias(
    state_dict: dict,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    bg_suppress_bias = float(state_dict.get('bg_suppress_bias', -10000.0))
    if bg_suppress_bias >= 0.0:
        return None

    mode = state_dict.get('mode', 'noop')

    if mode == 'write':
        if not state_dict.get('isolate_reference', True):
            return None
        write_masks = state_dict.get('write_fg_masks')
        if write_masks is None:
            return None
        return _build_spatial_isolation_bias(write_masks, seq_len, bg_suppress_bias, dtype, device)

    tgt_mask_2d = state_dict.get('tgt_fg_tensor')
    if tgt_mask_2d is None:
        return None

    cache = state_dict.setdefault('iso_bias_cache', {})
    cache_key = (seq_len, dtype, bg_suppress_bias)
    if cache_key in cache:
        return cache[cache_key]

    attn_bias = _build_spatial_isolation_bias(tgt_mask_2d, seq_len, bg_suppress_bias, dtype, device)
    cache[cache_key] = attn_bias
    return attn_bias


def _isolated_self_attention(
    attn,
    hidden_states: torch.Tensor,
    state_dict: dict,
    temb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    '''
    Runs self-attention while suppressing attention from foreground queries (inside mask)
    to background keys (outside mask) so background textures do not bleed into the object.
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

    attn_bias = _get_isolation_bias(state_dict, seq_len, q.dtype, q.device)

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
    foreground queries from pulling background texture across the mask boundary.
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
        if self.state_dict['mode'] in ('noop', 'read', 'write'):
            return _isolated_self_attention(attn, hidden_states, self.state_dict, temb=temb)
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
    Multi-view MasaCtrl attention processor with foreground token compaction,
    Best-View Routing, and bi-directional background-isolated self-attention.
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
        record_scores: bool = False,
    ) -> int:
        n_views = k_bank_h.shape[0]
        if n_views == 1 or not tgt_fg_flat.any():
            return 0

        q_fg = F.normalize(q_h[:, :, tgt_fg_flat, :].float(), dim=-1)  # [1, heads, L_fg, head_dim]
        scores = torch.full((n_views,), -1e9, device=q_h.device, dtype=torch.float32)

        for i in range(n_views):
            v_cols = valid_mask[i]
            if not v_cols.any():
                continue
            k_i = F.normalize(k_bank_h[i:i + 1, :, v_cols, :].float(), dim=-1)
            sim = torch.matmul(q_fg, k_i.transpose(-1, -2))  # [1, heads, L_fg, M_i]
            scores[i] = sim.max(dim=-1).values.mean()

        if record_scores:
            self.state_dict['view_scores'].append(scores.detach())

        return int(torch.argmax(scores).item())

    def _masked_mutual_branch(
        self,
        attn,
        hidden_states_1d: torch.Tensor,
        ref_bank: torch.Tensor,     # [N, M_max, C]
        valid_mask: torch.Tensor,   # [N, M_max]
        tgt_blend: torch.Tensor,    # [1, seq_len, 1]
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
            return _isolated_self_attention(attn, hidden_states, self.state_dict, temb=temb)

        if mode == 'read' and self.layer_name in self.state_dict['bank']:
            ref_bank, valid_mask = self.state_dict['bank'][self.layer_name]
            ref_bank = ref_bank.to(device=hidden_states.device, dtype=hidden_states.dtype)
            valid_mask = valid_mask.to(device=hidden_states.device)
            tgt_blend = self._get_tgt_blend(hidden_states.shape[1], hidden_states.dtype)

            # CFG batch size = 2: [uncond (0), cond (1)]
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
                        attn, hidden_states[:1], self.state_dict, temb=temb
                    )
                    out_uncond = style_fidelity * out_uncond_ref + (1.0 - style_fidelity) * out_uncond_self
                else:
                    out_uncond = _isolated_self_attention(
                        attn, hidden_states[:1], self.state_dict, temb=temb
                    )
                return torch.cat([out_uncond, out_cond], dim=0)

        # Phase 1 ('noop'): suppress background -> foreground attention while geometry forms
        return _isolated_self_attention(attn, hidden_states, self.state_dict, temb=temb)


class FastMultiViewMasaCtrlInpainter:
    '''
    Accelerated zero-shot multi-reference SD 1.5 Inpainting pipeline using:
      1. Closed-form forward noising q(z_t | z_0) instead of N-view DDIM inversion.
      2. Coarse-to-fine view pruning at switch_step down to top_k_views.
      3. Temporal KV caching across write_stride denoising steps.
      4. Bi-directional foreground-isolated self-attention on both target and reference views.
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

        self.state = {
            'mode': 'noop',
            'bank': {},
            'bank_chunks': {},
            'write_fg_masks': None,
            'tgt_fg_tensor': None,
            'bg_suppress_bias': -10000.0,
            'isolate_reference': True,
            'iso_bias_cache': {},
            'style_fidelity': 0.5,
            'record_view_scores': False,
            'view_scores': [],
        }
        self._register_attention_processors(hook_blocks)

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
    def generate(
        self,
        init_image: Image.Image,
        inpaint_mask: Image.Image,
        reference_image: Union[Image.Image, List[Image.Image]],
        reference_mask: Union[Image.Image, List[Image.Image]],
        attn_mask: Optional[Image.Image] = None,
        bg_suppress_bias: float = -10000.0,
        isolate_reference: bool = True,
        prompt: str = 'extreme macro closeup photograph of a single human finger, shadows',
        negative_prompt: str = '',
        width: int = 512,
        height: int = 512,
        angles_deg: Tuple[float, ...] = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0),
        scales: Tuple[float, ...] = (0.80, 1.00, 1.25),
        top_k_views: int = 2,
        write_stride: int = 2,
        num_inference_steps: int = 35,
        switch_step: int = 10,
        guidance_scale: float = 6.5,
        style_fidelity: Union[float, Callable[[int, int], float]] = _default_style_fidelity,
        seed: int = 42,
    ) -> Image.Image:
        self.state['bank'].clear()
        self.state['bank_chunks'].clear()
        self.state['view_scores'].clear()
        self.state['iso_bias_cache'].clear()
        self.state['record_view_scores'] = False
        self.state['bg_suppress_bias'] = float(bg_suppress_bias)
        self.state['isolate_reference'] = bool(isolate_reference)

        ref_images = [reference_image] if isinstance(reference_image, Image.Image) else list(reference_image)
        ref_masks = [reference_mask] if isinstance(reference_mask, Image.Image) else list(reference_mask)
        if len(ref_images) != len(ref_masks) or len(ref_images) == 0:
            raise ValueError('reference_image and reference_mask must be non-empty and have matching lengths')

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

        # 2. Precompute rotation & scale augmented views across all reference images
        aug_images: List[Image.Image] = []
        aug_masks: List[Image.Image] = []
        for ref_img, ref_msk in zip(ref_images, ref_masks):
            imgs_i, msks_i = _build_augmented_views(
                reference_image=ref_img,
                reference_mask=ref_msk,
                width=width,
                height=height,
                angles_deg=angles_deg,
                scales=scales,
            )
            aug_images.extend(imgs_i)
            aug_masks.extend(msks_i)
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

        # 3. Shared deterministic noise for closed-form q(z_t | z_0) forward trajectory
        noise_gen = torch.Generator(device=self.device).manual_seed(seed)
        base_noise = torch.randn(
            (1, ref_latents.shape[1], ref_latents.shape[2], ref_latents.shape[3]),
            generator=noise_gen,
            device=self.device,
            dtype=self.dtype,
        )

        # Mutable container for active views (pruned after switch_step)
        active = {
            'latents': ref_latents,
            'unet_masks': ref_unet_masks,
            'fg_tensors': ref_fg_tensors,
            'count': n_views,
        }

        null_embed, _ = self.pipe.encode_prompt(
            prompt='',
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )

        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps
        stride = max(1, int(write_stride))

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
            is_switch_step = step_idx == switch_step
            need_write = is_switch_step or ((step_idx - switch_step) % stride == 0)

            self.state['record_view_scores'] = is_switch_step and (active['count'] > top_k_views)
            if self.state['record_view_scores']:
                self.state['view_scores'].clear()

            if need_write:
                self.state['bank_chunks'].clear()
                t = timesteps[step_idx]
                cur_n = active['count']
                cur_latents = active['latents']
                cur_unet_masks = active['unet_masks']
                cur_fg_tensors = active['fg_tensors']

                self.state['mode'] = 'write'
                for start in range(0, cur_n, self.batch_chunk_size):
                    end = min(start + self.batch_chunk_size, cur_n)
                    b_chunk = end - start

                    self.state['write_fg_masks'] = cur_fg_tensors[start:end]
                    z_ref_t = self.pipe.scheduler.add_noise(
                        cur_latents[start:end],
                        base_noise.expand(b_chunk, -1, -1, -1),
                        t.expand(b_chunk),
                    )
                    z_scaled = self.pipe.scheduler.scale_model_input(z_ref_t, t)
                    unet_in = torch.cat(
                        [z_scaled, cur_unet_masks[start:end], cur_latents[start:end]], dim=1
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
            # Prune candidate bank down to top_k_views right after switch_step completes
            if step_index == switch_step and self.state['view_scores']:
                avg_scores = torch.stack(self.state['view_scores'], dim=0).mean(dim=0)
                k_keep = min(max(1, int(top_k_views)), active['count'])
                top_idx = torch.topk(avg_scores, k=k_keep, largest=True).indices

                active['latents'] = active['latents'][top_idx]
                active['unet_masks'] = active['unet_masks'][top_idx]
                active['fg_tensors'] = active['fg_tensors'][top_idx]
                active['count'] = k_keep

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
            self.state['iso_bias_cache'].clear()
            self.state['record_view_scores'] = False
            self.state['write_fg_masks'] = None

        return result


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt

    inpainter = FastMultiViewMasaCtrlInpainter(
        inpaint_model_id='sd-legacy/stable-diffusion-inpainting',
        device='cuda',
        hook_blocks=('up_blocks.2', 'up_blocks.3'),
        batch_chunk_size=6,
    )

    init_img = Image.open('assets/src.jpg')
    inpaint_msk = Image.open('assets/mask_rect.jpg')
    ref_imgs = [Image.open('references/1.jpeg')]
    ref_msks = [Image.open('references/1_mask.png')]

    # Optional: pass blurred mask to inpaint_mask and tight mask to attn_mask
    # blended_msk = inpaint_msk.filter(ImageFilter.MaxFilter(25)).filter(ImageFilter.GaussianBlur(45))

    def fidelity_func(s: int, n: int) -> float:
        return float(np.linspace(0.0, 0.5, n)[s])

    t0 = time.perf_counter()
    output = inpainter.generate(
        init_image=init_img,
        inpaint_mask=inpaint_msk,
        attn_mask=inpaint_msk,
        bg_suppress_bias=-10000.0,
        isolate_reference=True,
        reference_image=ref_imgs,
        reference_mask=ref_msks,
        angles_deg=(0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0),
        scales=(0.80, 1.00, 1.25),
        top_k_views=5,
        write_stride=2,
        num_inference_steps=35,
        switch_step=7,
        guidance_scale=6.5,
        style_fidelity=fidelity_func,
        seed=2,
    )
    elapsed = time.perf_counter() - t0
    print(f'Generated in {elapsed:.2f}s')

    output.save('output_fast_multiview_masactrl_sd15.png')
    plt.imshow(output)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
