import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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


def _default_bg_suppress_bias(step: int, num_steps: int) -> float:
    # Strong structural isolation early, relaxing to soft boundary blending (-2.5) late
    return float(np.linspace(-12.0, -2.5, num_steps)[step])


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


def _build_unit_isolation_weights(
    mask_2d: torch.Tensor,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    '''
    Builds a [B, 1, seq_len, seq_len] weight matrix m_i * (1 - m_j) in [0, 1]
    representing foreground-query to background-key suppression strength.
    '''
    b_mask, _, h_full, w_full = mask_2d.shape
    aspect = w_full / float(h_full)
    h_feat = max(1, int(round((seq_len / aspect) ** 0.5)))
    w_feat = max(1, seq_len // h_feat)

    mask_down = F.interpolate(mask_2d.float(), size=(h_feat, w_feat), mode='bilinear', align_corners=False)
    fg = mask_down.clamp(0.0, 1.0).view(b_mask, seq_len)

    weights = fg[:, :, None] * (1.0 - fg[:, None, :])  # [B, seq_len, seq_len]
    if not (weights > 1e-4).any():
        return None

    return weights[:, None, :, :].to(device=device, dtype=dtype)


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

        # Use precomputed unit weights after switch_step pruning for O(1) slicing
        if state_dict.get('cache_write_weights', False):
            all_write_masks = state_dict.get('all_active_fg_tensors')
            write_slice = state_dict.get('write_slice')
            if all_write_masks is not None and write_slice is not None:
                w_cache = state_dict.setdefault('write_unit_weight_cache', {})
                w_key = (seq_len, dtype)
                if w_key not in w_cache:
                    w_cache[w_key] = _build_unit_isolation_weights(
                        all_write_masks, seq_len, dtype, device
                    )
                full_weights = w_cache[w_key]
                if full_weights is None:
                    return None
                s0, s1 = write_slice
                return full_weights[s0:s1] * bg_suppress_bias

        unit_weights = _build_unit_isolation_weights(write_masks, seq_len, dtype, device)
        return None if unit_weights is None else unit_weights * bg_suppress_bias

    tgt_mask_2d = state_dict.get('tgt_fg_tensor')
    if tgt_mask_2d is None:
        return None

    cache = state_dict.setdefault('iso_weight_cache', {})
    cache_key = (seq_len, dtype)
    if cache_key not in cache:
        cache[cache_key] = _build_unit_isolation_weights(tgt_mask_2d, seq_len, dtype, device)

    unit_weights = cache[cache_key]
    return None if unit_weights is None else unit_weights * bg_suppress_bias


def _isolated_self_attention(
    attn,
    hidden_states: torch.Tensor,
    state_dict: dict,
    temb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    '''
    Runs self-attention while suppressing attention from foreground queries (inside mask)
    to background keys (outside mask) according to the current step's bias schedule.
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
    Applied to non-MasaCtrl self-attention layers (down_blocks, mid_block) to regulate
    cross-boundary attention between foreground and surrounding background context.
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
    Vectorized multi-view MasaCtrl attention processor with:
      - Chunked einsum view similarity scoring (zero per-view Python loops).
      - Cached projected (K_bank, V_bank) across write_stride steps and CFG branches.
      - Shared conditional-to-unconditional routing decisions.
      - Global-to-local spatial patch routing across candidate reference views.
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
        return tgt_down.clamp(0.0, 1.0).to(dtype=dtype).flatten(2).transpose(1, 2)  # [1, seq_len, 1]

    def _get_patch_grid_layout(
        self,
        seq_len: int,
        tgt_fg_flat: torch.Tensor,  # [seq_len] bool
        grid_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache = self.state_dict.setdefault('patch_grid_cache', {})
        cache_key = (seq_len, grid_size)
        if cache_key in cache:
            return cache[cache_key]

        tgt_mask_2d = self.state_dict['tgt_fg_tensor']
        _, _, h_full, w_full = tgt_mask_2d.shape
        aspect = w_full / float(h_full)
        h_feat = max(1, int(round((seq_len / aspect) ** 0.5)))
        w_feat = max(1, seq_len // h_feat)

        fg_2d = tgt_fg_flat.view(h_feat, w_feat)
        ys, xs = torch.where(fg_2d)
        y_min, y_max = int(ys.min().item()), int(ys.max().item())
        x_min, x_max = int(xs.min().item()), int(xs.max().item())

        y_span = max(1, y_max - y_min + 1)
        x_span = max(1, x_max - x_min + 1)
        gy = torch.clamp(((ys - y_min) * grid_size) // y_span, 0, grid_size - 1)
        gx = torch.clamp(((xs - x_min) * grid_size) // x_span, 0, grid_size - 1)
        cell_ids = (gy * grid_size + gx).to(device=device)  # [L_fg]
        fg_indices = torch.where(tgt_fg_flat)[0].to(device=device)

        cache[cache_key] = (fg_indices, cell_ids)
        return fg_indices, cell_ids

    def _get_projected_bank(
        self,
        attn,
        ref_bank: torch.Tensor,  # [K, M_max, C]
        head_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Projects ref_bank once per write update and reuses across stride & CFG passes.'''
        kv_cache = self.state_dict.setdefault('proj_kv_cache', {})
        if self.layer_name in kv_cache:
            return kv_cache[self.layer_name]

        k_views = ref_bank.shape[0]
        k_bank = attn.to_k(ref_bank)
        v_bank = attn.to_v(ref_bank)

        k_bank_h = k_bank.view(k_views, -1, attn.heads, head_dim).transpose(1, 2)
        v_bank_h = v_bank.view(k_views, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_k is not None:
            k_bank_h = attn.norm_k(k_bank_h)

        kv_cache[self.layer_name] = (k_bank_h, v_bank_h)
        return k_bank_h, v_bank_h

    def _compute_token_view_scores(
        self,
        q_h: torch.Tensor,          # [1, heads, seq_len, head_dim]
        k_bank_h: torch.Tensor,     # [K, heads, M_max, head_dim]
        valid_mask: torch.Tensor,   # [K, M_max]
        fg_indices: torch.Tensor,   # [L_fg]
        q_chunk_size: int = 512,
    ) -> torch.Tensor:
        '''
        Computes [K, L_fg] cosine similarity scores in vectorized query chunks
        without per-view Python loops or large transient similarity buffers.
        '''
        k_views = k_bank_h.shape[0]
        l_fg = int(fg_indices.shape[0])
        q_fg = F.normalize(q_h[:, :, fg_indices, :], dim=-1)  # [1, heads, L_fg, head_dim]
        k_norm = F.normalize(k_bank_h, dim=-1)                # [K, heads, M_max, head_dim]
        invalid_cols = ~valid_mask[:, None, None, :]          # [K, 1, 1, M_max]

        token_scores = torch.empty((k_views, l_fg), device=q_h.device, dtype=torch.float32)
        for q_start in range(0, l_fg, q_chunk_size):
            q_end = min(q_start + q_chunk_size, l_fg)
            sim = torch.einsum('ihqd,khmd->khqm', q_fg[:, :, q_start:q_end, :], k_norm)
            sim = sim.masked_fill(invalid_cols, -1e4)
            token_scores[:, q_start:q_end] = sim.max(dim=-1).values.float().mean(dim=1)

        return token_scores

    def _route_and_attend_fg(
        self,
        q_h: torch.Tensor,          # [1, heads, seq_len, head_dim]
        k_bank_h: torch.Tensor,     # [K, heads, M_max, head_dim]
        v_bank_h: torch.Tensor,     # [K, heads, M_max, head_dim]
        valid_mask: torch.Tensor,   # [K, M_max]
        tgt_fg_flat: torch.Tensor,  # [seq_len] bool
        use_patch_routing: bool,
        grid_size: int,
        record_scores: bool,
        cached_route: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        _, n_heads, seq_len, head_dim = q_h.shape
        k_views = k_bank_h.shape[0]
        inner_dim = n_heads * head_dim

        if k_views == 1 or not tgt_fg_flat.any():
            v_cols = valid_mask[0]
            f_obj = F.scaled_dot_product_attention(
                q_h,
                k_bank_h[0:1, :, v_cols, :],
                v_bank_h[0:1, :, v_cols, :],
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            return f_obj.transpose(1, 2).reshape(1, seq_len, inner_dim), {'type': 'global', 'best_idx': 0}

        # Reuse conditional branch's winning route during unconditional CFG pass
        if cached_route is not None:
            if cached_route['type'] == 'global':
                best_idx = cached_route['best_idx']
                v_cols = valid_mask[best_idx]
                f_obj = F.scaled_dot_product_attention(
                    q_h,
                    k_bank_h[best_idx:best_idx + 1, :, v_cols, :],
                    v_bank_h[best_idx:best_idx + 1, :, v_cols, :],
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                )
                return f_obj.transpose(1, 2).reshape(1, seq_len, inner_dim), cached_route

            fg_indices = cached_route['fg_indices']
            cell_ids = cached_route['cell_ids']
            best_per_cell = cached_route['best_per_cell']
            n_cells = len(best_per_cell)

            f_obj = torch.zeros((1, n_heads, seq_len, head_dim), device=q_h.device, dtype=q_h.dtype)
            for cell_idx in range(n_cells):
                in_cell = cell_ids == cell_idx
                if not in_cell.any():
                    continue
                q_indices = fg_indices[in_cell]
                v_best = int(best_per_cell[cell_idx].item())
                v_cols = valid_mask[v_best]
                f_obj[:, :, q_indices, :] = F.scaled_dot_product_attention(
                    q_h[:, :, q_indices, :],
                    k_bank_h[v_best:v_best + 1, :, v_cols, :],
                    v_bank_h[v_best:v_best + 1, :, v_cols, :],
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                ).to(dtype=q_h.dtype)
            return f_obj.transpose(1, 2).reshape(1, seq_len, inner_dim), cached_route

        fg_indices, cell_ids = self._get_patch_grid_layout(
            seq_len=seq_len,
            tgt_fg_flat=tgt_fg_flat,
            grid_size=max(1, grid_size),
            device=q_h.device,
        )

        token_scores = self._compute_token_view_scores(
            q_h=q_h,
            k_bank_h=k_bank_h,
            valid_mask=valid_mask,
            fg_indices=fg_indices,
        )

        if record_scores:
            self.state_dict['view_scores'].append(token_scores.mean(dim=-1).detach())

        if not use_patch_routing or grid_size <= 1:
            best_idx = int(torch.argmax(token_scores.mean(dim=-1)).item())
            v_cols = valid_mask[best_idx]
            f_obj = F.scaled_dot_product_attention(
                q_h,
                k_bank_h[best_idx:best_idx + 1, :, v_cols, :],
                v_bank_h[best_idx:best_idx + 1, :, v_cols, :],
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            route_info = {'type': 'global', 'best_idx': best_idx}
            return f_obj.transpose(1, 2).reshape(1, seq_len, inner_dim), route_info

        # Aggregate token scores into [K, grid_size, grid_size] patch scores
        n_cells = grid_size * grid_size
        patch_scores = torch.zeros((k_views, n_cells), device=q_h.device, dtype=torch.float32)
        patch_counts = torch.zeros((1, n_cells), device=q_h.device, dtype=torch.float32)

        patch_scores.scatter_add_(1, cell_ids.unsqueeze(0).expand(k_views, -1), token_scores)
        patch_counts.scatter_add_(
            1, cell_ids.unsqueeze(0), torch.ones_like(cell_ids, dtype=torch.float32).unsqueeze(0)
        )
        patch_scores = patch_scores / patch_counts.clamp(min=1.0)

        # Spatially smooth patch scores across neighboring grid cells to prevent abrupt view jumps
        grid_2d = patch_scores.view(1, k_views, grid_size, grid_size)
        kernel = torch.ones((k_views, 1, 3, 3), device=q_h.device, dtype=torch.float32)
        kernel[:, :, 1, 1] = 4.0
        smoothed = F.conv2d(F.pad(grid_2d, (1, 1, 1, 1), mode='replicate'), kernel, groups=k_views)
        patch_scores = smoothed.view(k_views, n_cells)

        best_per_cell = torch.argmax(patch_scores, dim=0)  # [n_cells]

        f_obj = torch.zeros((1, n_heads, seq_len, head_dim), device=q_h.device, dtype=q_h.dtype)
        for cell_idx in range(n_cells):
            in_cell = cell_ids == cell_idx
            if not in_cell.any():
                continue
            q_indices = fg_indices[in_cell]
            v_best = int(best_per_cell[cell_idx].item())
            v_cols = valid_mask[v_best]

            f_obj[:, :, q_indices, :] = F.scaled_dot_product_attention(
                q_h[:, :, q_indices, :],
                k_bank_h[v_best:v_best + 1, :, v_cols, :],
                v_bank_h[v_best:v_best + 1, :, v_cols, :],
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            ).to(dtype=q_h.dtype)

        route_info = {
            'type': 'patch',
            'fg_indices': fg_indices,
            'cell_ids': cell_ids,
            'best_per_cell': best_per_cell,
        }
        return f_obj.transpose(1, 2).reshape(1, seq_len, inner_dim), route_info

    def _masked_mutual_branch(
        self,
        attn,
        hidden_states_1d: torch.Tensor,
        ref_bank: torch.Tensor,     # [N, M_max, C]
        valid_mask: torch.Tensor,   # [N, M_max]
        tgt_blend: torch.Tensor,    # [1, seq_len, 1]
        record_scores: bool = False,
        cached_route: Optional[Dict[str, Any]] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
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

        inner_dim = k_tgt.shape[-1]
        head_dim = inner_dim // attn.heads

        q_h = q_tgt.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        k_tgt_h = k_tgt.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        v_tgt_h = v_tgt.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            q_h = attn.norm_q(q_h)
        if attn.norm_k is not None:
            k_tgt_h = attn.norm_k(k_tgt_h)

        k_bank_h, v_bank_h = self._get_projected_bank(attn, ref_bank, head_dim)

        tgt_fg_flat = tgt_blend[0, :, 0] > 0.5
        use_patch_routing = bool(self.state_dict.get('use_patch_routing', False))
        grid_size = int(self.state_dict.get('patch_grid_size', 1))

        f_obj, route_info = self._route_and_attend_fg(
            q_h=q_h,
            k_bank_h=k_bank_h,
            v_bank_h=v_bank_h,
            valid_mask=valid_mask,
            tgt_fg_flat=tgt_fg_flat,
            use_patch_routing=use_patch_routing,
            grid_size=grid_size,
            record_scores=record_scores,
            cached_route=cached_route,
        )
        f_obj = f_obj.to(q_tgt.dtype)

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

        return out / attn.rescale_output_factor, route_info

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
                out_cond, route_info = self._masked_mutual_branch(
                    attn,
                    hidden_states[1:],
                    ref_bank,
                    valid_mask,
                    tgt_blend,
                    record_scores=record_scores,
                    cached_route=None,
                    temb=temb,
                )
                style_fidelity = self.state_dict.get('style_fidelity', 0.5)
                if style_fidelity > 0.0:
                    out_uncond_ref, _ = self._masked_mutual_branch(
                        attn,
                        hidden_states[:1],
                        ref_bank,
                        valid_mask,
                        tgt_blend,
                        record_scores=False,
                        cached_route=route_info,
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

        return _isolated_self_attention(attn, hidden_states, self.state_dict, temb=temb)


class FastMultiViewMasaCtrlInpainter:
    '''
    Accelerated zero-shot multi-reference SD 1.5 Inpainting pipeline using:
      1. Closed-form forward noising q(z_t | z_0) instead of N-view DDIM inversion.
      2. Two-stage funnel pruning (switch_step -> top_k_views, funnel_step -> funnel_top_k_views).
      3. Vectorized einsum view scoring, cached write isolation biases, and cached KV projections.
      4. Scheduled bi-directional foreground-isolated self-attention.
      5. Global-to-local spatial patch routing across candidate reference views.
    '''
    def __init__(
        self,
        inpaint_model_id: str = 'sd-legacy/stable-diffusion-inpainting',
        device: str = 'cuda',
        hook_blocks: Tuple[str, ...] = ('up_blocks.2', 'up_blocks.3'),
        batch_chunk_size: int = 8,
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
            'proj_kv_cache': {},
            'write_fg_masks': None,
            'all_active_fg_tensors': None,
            'write_slice': None,
            'cache_write_weights': False,
            'write_unit_weight_cache': {},
            'tgt_fg_tensor': None,
            'bg_suppress_bias': -10000.0,
            'isolate_reference': True,
            'iso_weight_cache': {},
            'patch_grid_cache': {},
            'patch_grid_size': 2,
            'use_patch_routing': False,
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
        self.state['proj_kv_cache'].clear()
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
        bg_suppress_bias: Union[float, Callable[[int, int], float]] = _default_bg_suppress_bias,
        isolate_reference: bool = True,
        prompt: str = 'extreme macro closeup photograph of a single human finger, shadows',
        negative_prompt: str = '',
        width: int = 512,
        height: int = 512,
        angles_deg: Tuple[float, ...] = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0),
        scales: Tuple[float, ...] = (0.80, 1.00, 1.25),
        top_k_views: int = 12,
        funnel_top_k_views: int = 4,
        funnel_step: int = 10,
        patch_grid_size: int = 2,
        patch_start_step: int = 15,
        write_stride: int = 2,
        num_inference_steps: int = 35,
        switch_step: int = 1,
        guidance_scale: float = 6.5,
        style_fidelity: Union[float, Callable[[int, int], float]] = _default_style_fidelity,
        seed: int = 42,
    ) -> Image.Image:
        self.state['bank'].clear()
        self.state['bank_chunks'].clear()
        self.state['proj_kv_cache'].clear()
        self.state['view_scores'].clear()
        self.state['iso_weight_cache'].clear()
        self.state['write_unit_weight_cache'].clear()
        self.state['patch_grid_cache'].clear()
        self.state['record_view_scores'] = False
        self.state['use_patch_routing'] = False
        self.state['cache_write_weights'] = False
        self.state['patch_grid_size'] = max(1, int(patch_grid_size))
        self.state['isolate_reference'] = bool(isolate_reference)

        ref_images = [reference_image] if isinstance(reference_image, Image.Image) else list(reference_image)
        ref_masks = [reference_mask] if isinstance(reference_mask, Image.Image) else list(reference_mask)
        if len(ref_images) != len(ref_masks) or len(ref_images) == 0:
            raise ValueError('reference_image and reference_mask must be non-empty and have matching lengths')

        # 1. Prepare target image, blending mask, and foreground attention mask
        init_img = init_image.convert('RGB').resize((width, height), Image.LANCZOS)
        tgt_mask_arr = np.array(
            inpaint_mask.convert('L').resize((width, height), Image.BILINEAR), dtype=np.float32
        )
        if tgt_mask_arr.max() > 0.0 and tgt_mask_arr.max() < 200.0:
            tgt_mask_arr = np.clip((tgt_mask_arr / tgt_mask_arr.max()) * 255.0, 0.0, 255.0)
        tgt_mask_img = Image.fromarray(tgt_mask_arr.astype(np.uint8), mode='L')

        core_mask_src = attn_mask if attn_mask is not None else inpaint_mask
        core_arr = np.array(
            core_mask_src.convert('L').resize((width, height), Image.BILINEAR), dtype=np.float32
        ) / 255.0
        self.state['tgt_fg_tensor'] = torch.from_numpy(core_arr)[None, None].to(
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

        active = {
            'latents': ref_latents,
            'unet_masks': ref_unet_masks,
            'fg_tensors': ref_fg_tensors,
            'count': n_views,
            'force_write': False,
        }
        self.state['all_active_fg_tensors'] = ref_fg_tensors

        null_embed, _ = self.pipe.encode_prompt(
            prompt='',
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )

        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps
        stride = max(1, int(write_stride))
        enable_funnel = (
            funnel_top_k_views is not None
            and 1 <= int(funnel_top_k_views) < int(top_k_views)
            and int(funnel_step) > int(switch_step)
        )

        def resolve_style_fidelity(step_idx: int) -> float:
            val = float(style_fidelity(step_idx, len(timesteps))) if callable(style_fidelity) else float(style_fidelity)
            if not 0.0 <= val <= 1.0:
                raise ValueError('style_fidelity must be between 0.0 and 1.0')
            return val

        def resolve_bg_suppress_bias(step_idx: int) -> float:
            val = (
                float(bg_suppress_bias(step_idx, len(timesteps)))
                if callable(bg_suppress_bias)
                else float(bg_suppress_bias)
            )
            if val > 0.0:
                raise ValueError('bg_suppress_bias must be <= 0.0')
            return val

        def prune_active_views(k_target: int) -> None:
            if not self.state['view_scores']:
                return
            avg_scores = torch.stack(self.state['view_scores'], dim=0).mean(dim=0)
            k_keep = min(max(1, int(k_target)), active['count'])
            if k_keep >= active['count']:
                self.state['view_scores'].clear()
                return

            top_idx = torch.topk(avg_scores, k=k_keep, largest=True).indices
            active['latents'] = active['latents'][top_idx]
            active['unet_masks'] = active['unet_masks'][top_idx]
            active['fg_tensors'] = active['fg_tensors'][top_idx]
            active['count'] = k_keep
            active['force_write'] = True

            self.state['all_active_fg_tensors'] = active['fg_tensors']
            self.state['write_unit_weight_cache'].clear()
            self.state['proj_kv_cache'].clear()
            self.state['view_scores'].clear()

        def arm_step(step_idx: int) -> None:
            if step_idx >= len(timesteps):
                self.state['mode'] = 'noop'
                self.state['record_view_scores'] = False
                self.state['use_patch_routing'] = False
                return

            self.state['bg_suppress_bias'] = resolve_bg_suppress_bias(step_idx)

            if step_idx < switch_step:
                self.state['mode'] = 'noop'
                self.state['record_view_scores'] = False
                self.state['use_patch_routing'] = False
                return

            self.state['style_fidelity'] = resolve_style_fidelity(step_idx)
            self.state['use_patch_routing'] = (
                step_idx >= patch_start_step and self.state['patch_grid_size'] > 1
            )

            is_switch_step = step_idx == switch_step
            is_funnel_step = enable_funnel and (step_idx == int(funnel_step))
            need_write = (
                is_switch_step
                or active['force_write']
                or ((step_idx - switch_step) % stride == 0)
            )
            active['force_write'] = False

            # Enable cached write masks once initial N-view pool is pruned to <= 16 views
            self.state['cache_write_weights'] = (not is_switch_step) and (active['count'] <= 16)

            should_record = (
                (is_switch_step and active['count'] > top_k_views)
                or (is_funnel_step and active['count'] > int(funnel_top_k_views))
            )
            self.state['record_view_scores'] = should_record
            if should_record:
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
                    self.state['write_slice'] = (start, end)
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
                self.state['write_slice'] = None
                self._finalize_step_bank()

            self.state['mode'] = 'read'

        arm_step(0)

        def step_callback(pipe_obj, step_index, timestep, callback_kwargs):
            if step_index == switch_step and self.state['view_scores']:
                prune_active_views(top_k_views)
                self.state['record_view_scores'] = False
            elif enable_funnel and step_index == int(funnel_step) and self.state['view_scores']:
                prune_active_views(int(funnel_top_k_views))
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
            self.state['proj_kv_cache'].clear()
            self.state['view_scores'].clear()
            self.state['iso_weight_cache'].clear()
            self.state['write_unit_weight_cache'].clear()
            self.state['patch_grid_cache'].clear()
            self.state['record_view_scores'] = False
            self.state['use_patch_routing'] = False
            self.state['cache_write_weights'] = False
            self.state['all_active_fg_tensors'] = None
            self.state['write_fg_masks'] = None
            self.state['write_slice'] = None

        return result


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt

    inpainter = FastMultiViewMasaCtrlInpainter(
        inpaint_model_id='sd-legacy/stable-diffusion-inpainting',
        device='cuda',
        hook_blocks=('up_blocks.2', 'up_blocks.3'),
        batch_chunk_size=8,
    )

    init_img = Image.open('assets/src.jpg')
    inpaint_msk = Image.open('assets/mask_rect.jpg').filter(ImageFilter.MaxFilter(25))
    ref_imgs = [Image.open('references/1.jpeg')]
    ref_msks = [Image.open('references/1_mask.png')]

    attn_msk = inpaint_msk.filter(ImageFilter.GaussianBlur(3))
    blured_mask = inpaint_msk.filter(ImageFilter.MaxFilter(25)).filter(ImageFilter.GaussianBlur(45))

    def fidelity_func(s: int, n: int) -> float:
        return float(np.linspace(0.0, 0.5, n)[s])

    def bias_func(s: int, n: int) -> float:
        return float(np.linspace(-12.0, -2.5, n)[s])

    t0 = time.perf_counter()
    output = inpainter.generate(
        init_image=init_img,
        inpaint_mask=blured_mask,
        attn_mask=attn_msk,
        bg_suppress_bias=bias_func,
        isolate_reference=True,
        reference_image=ref_imgs,
        reference_mask=ref_msks,
        angles_deg=tuple(np.linspace(-15.0, 15.0, 8)),
        scales=tuple(np.linspace(0.70, 1.3, 4)),
        top_k_views=15,
        funnel_top_k_views=4,
        funnel_step=10,
        patch_grid_size=2,
        patch_start_step=15,
        write_stride=2,
        num_inference_steps=35,
        switch_step=1,
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
