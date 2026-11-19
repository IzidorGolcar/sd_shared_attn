"""Attention processors used by the multi-reference inpainter."""
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0

from .reference_utils import extract_foreground_tokens_ragged

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
      - Unpadded ragged foreground token extraction during write.
      - Two-axis chunked einsum view similarity scoring (bounded peak VRAM for 100+ views).
      - Cached projected (K_bank, V_bank) with immediate release of raw ref_bank after projection.
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
        ref_bank: Optional[torch.Tensor],  # [K, M_max, C] or None if already projected
        valid_mask: torch.Tensor,
        head_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Projects ref_bank once per write update, releases raw ref_bank from self.state['bank']
        to avoid holding 3x copies on GPU, and reuses (k_bank_h, v_bank_h) across CFG & stride.
        '''
        kv_cache = self.state_dict.setdefault('proj_kv_cache', {})
        if self.layer_name in kv_cache:
            return kv_cache[self.layer_name]

        if ref_bank is None:
            raise RuntimeError(f'Missing unprojected ref_bank for {self.layer_name}')

        k_views = ref_bank.shape[0]
        k_bank = attn.to_k(ref_bank)
        v_bank = attn.to_v(ref_bank)

        k_bank_h = k_bank.view(k_views, -1, attn.heads, head_dim).transpose(1, 2)
        v_bank_h = v_bank.view(k_views, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_k is not None:
            k_bank_h = attn.norm_k(k_bank_h)

        kv_cache[self.layer_name] = (k_bank_h, v_bank_h)
        # Drop raw unprojected tensor from bank so GPU holds only projected K/V
        if self.layer_name in self.state_dict['bank']:
            self.state_dict['bank'][self.layer_name] = (None, valid_mask)
        return k_bank_h, v_bank_h

    def _compute_token_view_scores(
        self,
        q_h: torch.Tensor,          # [1, heads, seq_len, head_dim]
        k_bank_h: torch.Tensor,     # [K, heads, M_max, head_dim]
        valid_mask: torch.Tensor,   # [K, M_max]
        fg_indices: torch.Tensor,   # [L_fg]
        q_chunk_size: int = 256,
        k_chunk_size: int = 8,
    ) -> torch.Tensor:
        '''
        Computes [K, L_fg] cosine similarity scores in bounded (k_chunk_size, q_chunk_size)
        tiles with in-place masking so peak transient VRAM stays under ~80 MB even for 100+ views.
        '''
        k_views = k_bank_h.shape[0]
        l_fg = int(fg_indices.shape[0])
        q_fg = F.normalize(q_h[:, :, fg_indices, :], dim=-1)  # [1, heads, L_fg, head_dim]

        token_scores = torch.empty((k_views, l_fg), device=q_h.device, dtype=torch.float32)

        for k_start in range(0, k_views, k_chunk_size):
            k_end = min(k_start + k_chunk_size, k_views)
            k_norm = F.normalize(k_bank_h[k_start:k_end], dim=-1)          # [k_c, heads, M_max, head_dim]
            invalid_cols = ~valid_mask[k_start:k_end, None, None, :]       # [k_c, 1, 1, M_max]

            for q_start in range(0, l_fg, q_chunk_size):
                q_end = min(q_start + q_chunk_size, l_fg)
                sim = torch.einsum('ihqd,khmd->khqm', q_fg[:, :, q_start:q_end, :], k_norm)
                sim.masked_fill_(invalid_cols, -1e4)
                token_scores[k_start:k_end, q_start:q_end] = sim.max(dim=-1).values.float().mean(dim=1)
                del sim

            del k_norm

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

        n_cells = grid_size * grid_size
        patch_scores = torch.zeros((k_views, n_cells), device=q_h.device, dtype=torch.float32)
        patch_counts = torch.zeros((1, n_cells), device=q_h.device, dtype=torch.float32)

        patch_scores.scatter_add_(1, cell_ids.unsqueeze(0).expand(k_views, -1), token_scores)
        patch_counts.scatter_add_(
            1, cell_ids.unsqueeze(0), torch.ones_like(cell_ids, dtype=torch.float32).unsqueeze(0)
        )
        patch_scores = patch_scores / patch_counts.clamp(min=1.0)

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
        ref_bank: Optional[torch.Tensor],
        valid_mask: torch.Tensor,
        tgt_blend: torch.Tensor,
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

        k_bank_h, v_bank_h = self._get_projected_bank(attn, ref_bank, valid_mask, head_dim)

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
            skip_prefixes: Tuple[str, ...] = self.state_dict.get('write_skip_prefixes', ())
            if not (skip_prefixes and self.layer_name.startswith(skip_prefixes)):
                write_masks = self.state_dict.get('write_fg_masks')
                if write_masks is not None and not torch.isnan(hidden_states).any():
                    ragged_views = extract_foreground_tokens_ragged(
                        hidden_states.detach(), write_masks
                    )
                    self.state_dict['bank_chunks'].setdefault(self.layer_name, []).extend(
                        ragged_views
                    )
            return _isolated_self_attention(attn, hidden_states, self.state_dict, temb=temb)

        if mode == 'read' and self.layer_name in self.state_dict['bank']:
            ref_bank, valid_mask = self.state_dict['bank'][self.layer_name]
            if ref_bank is not None:
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


