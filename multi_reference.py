import gc
import math
import os
import shutil
import tempfile
import weakref
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from diffusers import (
    DDIMScheduler,
    StableDiffusionInpaintPipeline,
)
from diffusers.models.attention_processor import AttnProcessor2_0


def _flush_cuda_memory() -> None:
    '''Releases unreferenced Python/PyTorch tensors and defragments the CUDA caching allocator.'''
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_available_ram_bytes() -> int:
    '''Returns currently available system RAM in bytes without requiring external dependencies.'''
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().available)
    except ImportError:
        pass

    if os.path.exists('/proc/meminfo'):
        try:
            with open('/proc/meminfo', 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        return int(line.split()[1]) * 1024
        except OSError:
            pass

    try:
        pages = os.sysconf('SC_AVPHYS_PAGES')
        page_size = os.sysconf('SC_PAGE_SIZE')
        if pages > 0 and page_size > 0:
            return int(pages * page_size)
    except (ValueError, OSError, AttributeError):
        pass

    return 8 * (1024 ** 3)


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


def extract_foreground_tokens_ragged(
    hidden_states: torch.Tensor,  # [B, seq_len, C]
    fg_mask_2d: torch.Tensor,     # [B, 1, H, W]
) -> List[torch.Tensor]:
    '''
    Extracts unpadded per-view foreground token tensors [M_i, C] so small-scale views
    never waste RAM or SSD space on zero-padding up to M_max.
    '''
    b_size, seq_len, _ = hidden_states.shape
    _, _, h_full, w_full = fg_mask_2d.shape
    aspect = w_full / float(h_full)
    h_feat = max(1, int(round((seq_len / aspect) ** 0.5)))
    w_feat = max(1, seq_len // h_feat)

    mask_flat = (
        F.interpolate(fg_mask_2d.float(), size=(h_feat, w_feat), mode='nearest')
        .flatten(1) > 0.5
    )  # [B, seq_len]

    views: List[torch.Tensor] = []
    for i in range(b_size):
        idx = torch.where(mask_flat[i])[0]
        if len(idx) == 0:
            idx = torch.arange(min(seq_len, 1), device=hidden_states.device)
        views.append(hidden_states[i, idx].contiguous())

    return views


class TieredReferenceCache:
    '''
    Three-tier memory-aware storage manager for precomputed multi-view reference banks:
      - Tier 1 ('pinned'): Pinned host RAM up to max_pinned_bytes for fastest PCIe DMA.
      - Tier 2 ('ram'): Standard pageable host RAM while available RAM > min_free_ram_bytes.
      - Tier 3 ('ssd'): Zero-copy OS page-cached numpy.memmap files on SSD when RAM runs low.

    Stores unpadded ragged 1D token arrays with optional per-token int8 quantization,
    eliminating zero-padding bloat and paging in only the surviving pruned views from SSD.
    '''
    def __init__(
        self,
        max_pinned_bytes: int = 2 * (1024 ** 3),
        min_free_ram_bytes: int = 4 * (1024 ** 3),
        quantize: bool = True,
        cache_dir: Optional[str] = None,
        enable_pinned: bool = True,
    ):
        self.max_pinned_bytes = max(0, int(max_pinned_bytes))
        self.min_free_ram_bytes = max(512 * (1024 ** 2), int(min_free_ram_bytes))
        self.quantize = bool(quantize)
        self.cache_dir = cache_dir
        self.enable_pinned = bool(enable_pinned)

        self.pinned_bytes_used = 0
        self.tier_bytes: Dict[str, int] = {'pinned': 0, 'ram': 0, 'ssd': 0}
        self.entries: Dict[Tuple[int, str], Dict[str, Any]] = {}
        self.available_steps: set = set()

        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None
        self._finalizer: Optional[weakref.finalize] = None
        self._memmaps: List[np.memmap] = []
        self._file_counter = 0

    @staticmethod
    def _cleanup_temp_dir(temp_dir: Optional[tempfile.TemporaryDirectory]) -> None:
        if temp_dir is not None:
            try:
                temp_dir.cleanup()
            except OSError:
                pass

    def _ensure_temp_dir(self) -> str:
        if self._temp_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(
                prefix='masactrl_ref_cache_',
                dir=self.cache_dir,
            )
            self._finalizer = weakref.finalize(self, self._cleanup_temp_dir, self._temp_dir)
        return self._temp_dir.name

    def _write_memmap(self, arr_cpu: torch.Tensor, tag: str) -> np.memmap:
        tmp_path = self._ensure_temp_dir()
        np_arr = arr_cpu.numpy()
        needed_bytes = int(np_arr.nbytes)

        free_disk = shutil.disk_usage(tmp_path).free
        if free_disk < needed_bytes + 256 * (1024 ** 2):
            raise RuntimeError(
                f'Insufficient disk space in {tmp_path} for SSD reference cache '
                f'(needed {needed_bytes / (1024 ** 2):.1f} MB, free {free_disk / (1024 ** 2):.1f} MB).'
            )

        self._file_counter += 1
        file_path = os.path.join(tmp_path, f'bank_{self._file_counter:05d}_{tag}.dat')
        mm = np.memmap(file_path, dtype=np_arr.dtype, mode='w+', shape=np_arr.shape)
        mm[:] = np_arr[:]
        mm.flush()
        del mm

        mm_ro = np.memmap(file_path, dtype=np_arr.dtype, mode='r+', shape=np_arr.shape)
        self._memmaps.append(mm_ro)
        return mm_ro

    def put_layer(
        self,
        step_idx: int,
        layer_name: str,
        view_tensors: List[torch.Tensor],  # List of N tensors, each [M_i, C] on GPU
    ) -> None:
        lengths = [int(t.shape[0]) for t in view_tensors]
        offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])

        packed_gpu = torch.cat(view_tensors, dim=0)  # [Total_FG, C]
        channels = int(packed_gpu.shape[1])

        if self.quantize:
            scales_gpu = (
                packed_gpu.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 127.0
            ).to(dtype=torch.float16)
            q_gpu = torch.round(packed_gpu / scales_gpu).clamp(-127, 127).to(dtype=torch.int8)
            q_cpu = q_gpu.to('cpu')
            scales_cpu: Optional[torch.Tensor] = scales_gpu.to('cpu')
            entry_bytes = int(q_cpu.numel() * q_cpu.element_size() + scales_cpu.numel() * scales_cpu.element_size())
        else:
            q_cpu = packed_gpu.to('cpu')
            scales_cpu = None
            entry_bytes = int(q_cpu.numel() * q_cpu.element_size())

        avail_ram = _get_available_ram_bytes()
        can_use_ram = (avail_ram - entry_bytes) > self.min_free_ram_bytes

        if can_use_ram and self.enable_pinned and (self.pinned_bytes_used + entry_bytes <= self.max_pinned_bytes):
            tier = 'pinned'
            data_ref: Any = q_cpu.pin_memory()
            scale_ref: Any = scales_cpu.pin_memory() if scales_cpu is not None else None
            self.pinned_bytes_used += entry_bytes
        elif can_use_ram:
            tier = 'ram'
            data_ref = q_cpu
            scale_ref = scales_cpu
        else:
            tier = 'ssd'
            safe_name = layer_name.replace('.', '_')
            data_ref = self._write_memmap(q_cpu, f's{step_idx}_{safe_name}_tok')
            scale_ref = (
                self._write_memmap(scales_cpu, f's{step_idx}_{safe_name}_scl')
                if scales_cpu is not None
                else None
            )

        self.tier_bytes[tier] += entry_bytes
        self.available_steps.add(int(step_idx))
        self.entries[(int(step_idx), layer_name)] = {
            'tier': tier,
            'quantized': self.quantize,
            'channels': channels,
            'offsets': offsets,
            'data': data_ref,
            'scales': scale_ref,
            'bytes': entry_bytes,
        }

    def fetch_step_layers(
        self,
        step_idx: int,
        view_indices: Optional[Iterable[int]],
        n_total_views: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, Tuple[Optional[torch.Tensor], torch.Tensor]]:
        '''
        Reconstructs padded [K, M_max_subset, C] GPU tensors for only the active view_indices.
        For SSD-spilled entries, only the byte ranges of active views are faulted in from disk.
        '''
        idx_list = list(range(n_total_views)) if view_indices is None else [int(i) for i in view_indices]
        k_count = len(idx_list)
        result: Dict[str, Tuple[Optional[torch.Tensor], torch.Tensor]] = {}

        for (s_idx, layer_name), entry in self.entries.items():
            if s_idx != step_idx:
                continue

            offsets = entry['offsets']
            data_src = entry['data']
            scale_src = entry['scales']
            channels = entry['channels']
            is_ssd = entry['tier'] == 'ssd'

            lengths: List[int] = []
            tok_chunks: List[torch.Tensor] = []
            scl_chunks: List[torch.Tensor] = []

            for v in idx_list:
                s0, s1 = int(offsets[v]), int(offsets[v + 1])
                lengths.append(s1 - s0)
                if is_ssd:
                    tok_chunks.append(torch.from_numpy(np.array(data_src[s0:s1], copy=True)))
                    if scale_src is not None:
                        scl_chunks.append(torch.from_numpy(np.array(scale_src[s0:s1], copy=True)))
                else:
                    tok_chunks.append(data_src[s0:s1])
                    if scale_src is not None:
                        scl_chunks.append(scale_src[s0:s1])

            m_max = max(max(lengths), 1)
            cat_tok_cpu = torch.cat(tok_chunks, dim=0)
            cat_tok_gpu = cat_tok_cpu.to(device=device, non_blocking=True)

            if entry['quantized']:
                cat_scl_cpu = torch.cat(scl_chunks, dim=0)
                cat_scl_gpu = cat_scl_cpu.to(device=device, dtype=dtype, non_blocking=True)
                cat_gpu = cat_tok_gpu.to(dtype=dtype) * cat_scl_gpu
            else:
                cat_gpu = cat_tok_gpu.to(dtype=dtype)

            full_comp = torch.zeros((k_count, m_max, channels), device=device, dtype=dtype)
            full_mask = torch.zeros((k_count, m_max), device=device, dtype=torch.bool)

            cursor = 0
            for out_i, length in enumerate(lengths):
                if length > 0:
                    full_comp[out_i, :length] = cat_gpu[cursor:cursor + length]
                    full_mask[out_i, :length] = True
                cursor += length

            result[layer_name] = (full_comp, full_mask)

        return result

    def stats(self) -> Dict[str, float]:
        total = sum(self.tier_bytes.values())
        return {
            'pinned_mb': round(self.tier_bytes['pinned'] / (1024 ** 2), 2),
            'ram_mb': round(self.tier_bytes['ram'] / (1024 ** 2), 2),
            'ssd_mb': round(self.tier_bytes['ssd'] / (1024 ** 2), 2),
            'total_mb': round(total / (1024 ** 2), 2),
            'num_entries': len(self.entries),
        }

    def clear(self) -> None:
        self.entries.clear()
        self.available_steps.clear()
        self.pinned_bytes_used = 0
        self.tier_bytes = {'pinned': 0, 'ram': 0, 'ssd': 0}

        if self._memmaps:
            for mm in self._memmaps:
                try:
                    if getattr(mm, '_mmap', None) is not None:
                        mm._mmap.close()
                except Exception:
                    pass
            self._memmaps.clear()
            gc.collect()

        if self._finalizer is not None and self._finalizer.alive:
            self._finalizer()
        elif self._temp_dir is not None:
            self._cleanup_temp_dir(self._temp_dir)
        self._temp_dir = None
        self._finalizer = None


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


class FastMultiViewMasaCtrlInpainter:
    '''
    Accelerated zero-shot multi-reference SD 1.5 Inpainting pipeline with:
      1. Smart 3-tier reference cache (Pinned RAM -> Pageable RAM -> Zero-Copy SSD memmap).
      2. Unpadded ragged 1D foreground token packing + optional int8 per-token quantization.
      3. Two-stage progressive funnel pruning (switch_step -> funnel_step).
      4. Bounded-VRAM chunked einsum view scoring, immediate post-projection bank release,
         and automatic CUDA cache defragmentation across repeated generate() calls.
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
        self.ref_cache: Optional[Dict[str, Any]] = None

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            inpaint_model_id,
            torch_dtype=self.dtype,
            use_safetensors=False,
            safety_checker=None,
        ).to(self.device)

        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_vae_slicing()

        self.state = {
            'mode': 'noop',
            'bank': {},
            'bank_chunks': {},
            'proj_kv_cache': {},
            'write_skip_prefixes': (),
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

    def get_cache_stats(self) -> Dict[str, float]:
        '''Returns RAM, Pinned RAM, and SSD memmap memory usage in MB for cached references.'''
        if self.ref_cache is None:
            return {'pinned_mb': 0.0, 'ram_mb': 0.0, 'ssd_mb': 0.0, 'total_mb': 0.0, 'num_entries': 0}
        return self.ref_cache['store'].stats()

    def clear_references(self) -> None:
        '''Frees cached multi-view reference banks from RAM and deletes any SSD memmap files.'''
        if self.ref_cache is not None and 'store' in self.ref_cache:
            self.ref_cache['store'].clear()
        self.ref_cache = None
        self.state['bank'].clear()
        self.state['bank_chunks'].clear()
        self.state['proj_kv_cache'].clear()
        self.state['write_unit_weight_cache'].clear()
        self.state['all_active_fg_tensors'] = None
        _flush_cuda_memory()

    @torch.no_grad()
    def set_references(
        self,
        reference_image: Union[Image.Image, List[Image.Image]],
        reference_mask: Union[Image.Image, List[Image.Image]],
        angles_deg: Tuple[float, ...] = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0),
        scales: Tuple[float, ...] = (0.80, 1.00, 1.25),
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 35,
        switch_step: int = 1,
        funnel_step: Optional[int] = 10,
        highres_start_step: int = 4,
        write_stride: int = 2,
        bg_suppress_bias: Union[float, Callable[[int, int], float]] = _default_bg_suppress_bias,
        isolate_reference: bool = True,
        ref_seed: int = 42,
        max_pinned_gb: float = 2.0,
        min_free_ram_gb: float = 4.0,
        quantize_cache: bool = True,
        cache_dir: Optional[str] = None,
    ) -> Dict[str, float]:
        '''
        Precomputes augmented views, VAE latents, and compacted foreground token banks
        into a 3-tier cache (Pinned RAM -> Pageable RAM -> Zero-Copy SSD memmap).
        '''
        self.clear_references()
        self.state['isolate_reference'] = bool(isolate_reference)

        ref_images = [reference_image] if isinstance(reference_image, Image.Image) else list(reference_image)
        ref_masks = [reference_mask] if isinstance(reference_mask, Image.Image) else list(reference_mask)
        if len(ref_images) != len(ref_masks) or len(ref_images) == 0:
            raise ValueError('reference_image and reference_mask must be non-empty and have matching lengths')

        store = TieredReferenceCache(
            max_pinned_bytes=int(max_pinned_gb * (1024 ** 3)),
            min_free_ram_bytes=int(min_free_ram_gb * (1024 ** 3)),
            quantize=quantize_cache,
            cache_dir=cache_dir,
            enable_pinned=(self.device.type == 'cuda'),
        )

        # 1. Build rotation & scale augmented views across all reference images
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

        ref_fg_tensors = torch.from_numpy(
            np.stack([(np.array(m, dtype=np.uint8) > 127).astype(np.float32) for m in aug_masks], axis=0)
        )[:, None].to(device=self.device, dtype=self.dtype)  # [N, 1, H, W]

        # Encode in chunks directly from PIL -> tensor so full N-image pixel tensor is never allocated on GPU
        latent_chunks = []
        for start in range(0, n_views, self.batch_chunk_size):
            end = min(start + self.batch_chunk_size, n_views)
            batch_px = torch.cat(
                [
                    self.pipe.image_processor.preprocess(img, height=height, width=width)
                    for img in aug_images[start:end]
                ],
                dim=0,
            ).to(device=self.device, dtype=self.dtype)
            lat = self.pipe.vae.encode(batch_px).latent_dist.mean
            latent_chunks.append((lat * self.pipe.vae.config.scaling_factor).to(dtype=self.dtype))
            del batch_px, lat

        ref_latents = torch.cat(latent_chunks, dim=0)
        del latent_chunks
        ref_unet_masks = F.interpolate(ref_fg_tensors, size=(height // 8, width // 8), mode='nearest')

        # 2. Deterministic noise trajectory for closed-form forward noising q(z_t | z_0)
        noise_gen = torch.Generator(device=self.device).manual_seed(ref_seed)
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

        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps
        stride = max(1, int(write_stride))

        write_steps_set = set(range(int(switch_step), len(timesteps), stride))
        if funnel_step is not None and int(switch_step) < int(funnel_step) < len(timesteps):
            write_steps_set.add(int(funnel_step))
        if int(switch_step) < int(highres_start_step) < len(timesteps):
            write_steps_set.add(int(highres_start_step))
        write_steps = sorted(write_steps_set)

        self.state['all_active_fg_tensors'] = ref_fg_tensors
        self.state['cache_write_weights'] = n_views <= 24

        try:
            for step_idx in write_steps:
                bias_val = (
                    float(bg_suppress_bias(step_idx, len(timesteps)))
                    if callable(bg_suppress_bias)
                    else float(bg_suppress_bias)
                )
                if bias_val > 0.0:
                    raise ValueError('bg_suppress_bias must be <= 0.0')
                self.state['bg_suppress_bias'] = bias_val
                self.state['bank_chunks'].clear()

                self.state['write_skip_prefixes'] = (
                    ('up_blocks.3',) if step_idx < int(highres_start_step) else ()
                )

                t = timesteps[step_idx]
                self.state['mode'] = 'write'
                for start in range(0, n_views, self.batch_chunk_size):
                    end = min(start + self.batch_chunk_size, n_views)
                    b_chunk = end - start

                    self.state['write_fg_masks'] = ref_fg_tensors[start:end]
                    self.state['write_slice'] = (start, end)
                    z_ref_t = self.pipe.scheduler.add_noise(
                        ref_latents[start:end],
                        base_noise.expand(b_chunk, -1, -1, -1),
                        t.expand(b_chunk),
                    )
                    z_scaled = self.pipe.scheduler.scale_model_input(z_ref_t, t)
                    unet_in = torch.cat(
                        [z_scaled, ref_unet_masks[start:end], ref_latents[start:end]], dim=1
                    )
                    self.pipe.unet(
                        unet_in,
                        t,
                        encoder_hidden_states=null_embed.expand(b_chunk, -1, -1),
                        return_dict=False,
                    )
                    del z_ref_t, z_scaled, unet_in

                self.state['write_fg_masks'] = None
                self.state['write_slice'] = None

                for layer_name, view_list in self.state['bank_chunks'].items():
                    store.put_layer(step_idx, layer_name, view_list)
                self.state['bank_chunks'].clear()

            self.ref_cache = {
                'n_views': n_views,
                'width': width,
                'height': height,
                'num_inference_steps': num_inference_steps,
                'switch_step': int(switch_step),
                'funnel_step': None if funnel_step is None else int(funnel_step),
                'highres_start_step': int(highres_start_step),
                'write_stride': stride,
                'isolate_reference': bool(isolate_reference),
                'store': store,
            }
            return store.stats()
        except Exception:
            store.clear()
            raise
        finally:
            self.state['mode'] = 'noop'
            self.state['bank'].clear()
            self.state['bank_chunks'].clear()
            self.state['proj_kv_cache'].clear()
            self.state['write_unit_weight_cache'].clear()
            self.state['write_skip_prefixes'] = ()
            self.state['cache_write_weights'] = False
            self.state['all_active_fg_tensors'] = None
            self.state['write_fg_masks'] = None
            self.state['write_slice'] = None
            del ref_fg_tensors, ref_latents, ref_unet_masks, base_noise, null_embed
            _flush_cuda_memory()

    @torch.no_grad()
    def generate(
        self,
        init_image: Image.Image,
        inpaint_mask: Image.Image,
        reference_image: Optional[Union[Image.Image, List[Image.Image]]] = None,
        reference_mask: Optional[Union[Image.Image, List[Image.Image]]] = None,
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
        funnel_step: Optional[int] = 10,
        highres_start_step: int = 4,
        patch_grid_size: int = 2,
        patch_start_step: int = 15,
        write_stride: int = 2,
        num_inference_steps: int = 35,
        switch_step: int = 1,
        guidance_scale: float = 6.5,
        style_fidelity: Union[float, Callable[[int, int], float]] = _default_style_fidelity,
        seed: int = 42,
        ref_seed: Optional[int] = None,
        max_pinned_gb: float = 2.0,
        min_free_ram_gb: float = 4.0,
        quantize_cache: bool = True,
        cache_dir: Optional[str] = None,
    ) -> Image.Image:
        # Build reference cache on demand if reference_image was passed directly
        if reference_image is not None:
            if reference_mask is None:
                raise ValueError('reference_mask must be provided when reference_image is passed')
            self.set_references(
                reference_image=reference_image,
                reference_mask=reference_mask,
                angles_deg=angles_deg,
                scales=scales,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                switch_step=switch_step,
                funnel_step=funnel_step,
                highres_start_step=highres_start_step,
                write_stride=write_stride,
                bg_suppress_bias=bg_suppress_bias,
                isolate_reference=isolate_reference,
                ref_seed=seed if ref_seed is None else ref_seed,
                max_pinned_gb=max_pinned_gb,
                min_free_ram_gb=min_free_ram_gb,
                quantize_cache=quantize_cache,
                cache_dir=cache_dir,
            )

        if self.ref_cache is None:
            raise RuntimeError('No references cached. Call set_references() first or pass reference_image/reference_mask.')

        if (self.ref_cache['width'], self.ref_cache['height']) != (width, height):
            raise ValueError(
                f'Cached references have resolution {(self.ref_cache["width"], self.ref_cache["height"])}, '
                f'but generate() was called with {(width, height)}.'
            )
        if self.ref_cache['num_inference_steps'] != num_inference_steps:
            raise ValueError(
                f'Cached references were built for num_inference_steps={self.ref_cache["num_inference_steps"]}, '
                f'but generate() was called with {num_inference_steps}.'
            )

        self.state['bank'].clear()
        self.state['bank_chunks'].clear()
        self.state['proj_kv_cache'].clear()
        self.state['view_scores'].clear()
        self.state['iso_weight_cache'].clear()
        self.state['patch_grid_cache'].clear()
        self.state['record_view_scores'] = False
        self.state['use_patch_routing'] = False
        self.state['patch_grid_size'] = max(1, int(patch_grid_size))
        _flush_cuda_memory()

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

        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps

        store: TieredReferenceCache = self.ref_cache['store']
        n_views = self.ref_cache['n_views']
        active = {
            'indices': None,  # None means all [0..n_views-1]; otherwise 1D LongTensor on CPU
            'count': n_views,
        }

        enable_funnel = (
            funnel_top_k_views is not None
            and funnel_step is not None
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

        def load_cached_step_bank(step_idx: int) -> None:
            self.state['bank'].clear()
            self.state['proj_kv_cache'].clear()
            idx_list = None if active['indices'] is None else active['indices'].tolist()
            self.state['bank'] = store.fetch_step_layers(
                step_idx=step_idx,
                view_indices=idx_list,
                n_total_views=n_views,
                device=self.device,
                dtype=self.dtype,
            )

        def prune_active_views(k_target: int) -> None:
            if not self.state['view_scores']:
                return
            avg_scores = torch.stack(self.state['view_scores'], dim=0).mean(dim=0)
            k_keep = min(max(1, int(k_target)), active['count'])
            if k_keep >= active['count']:
                self.state['view_scores'].clear()
                return

            top_local = torch.topk(avg_scores, k=k_keep, largest=True).indices
            top_local_cpu = top_local.to('cpu')

            if active['indices'] is None:
                active['indices'] = top_local_cpu
            else:
                active['indices'] = active['indices'][top_local_cpu]
            active['count'] = k_keep

            # Prune both projected KV cache and valid masks immediately, then release pruned GPU memory
            for layer_name, (k_bh, v_bh) in list(self.state['proj_kv_cache'].items()):
                self.state['proj_kv_cache'][layer_name] = (
                    k_bh[top_local].contiguous(),
                    v_bh[top_local].contiguous(),
                )
            for layer_name, (comp_gpu, valid_gpu) in list(self.state['bank'].items()):
                self.state['bank'][layer_name] = (
                    None if comp_gpu is None else comp_gpu[top_local].contiguous(),
                    valid_gpu[top_local].contiguous(),
                )

            self.state['view_scores'].clear()
            _flush_cuda_memory()

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

            if step_idx in store.available_steps:
                load_cached_step_bank(step_idx)

            should_record = (
                (is_switch_step and active['count'] > top_k_views)
                or (is_funnel_step and active['count'] > int(funnel_top_k_views))
            )
            self.state['record_view_scores'] = should_record
            if should_record:
                self.state['view_scores'].clear()

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
            self.state['patch_grid_cache'].clear()
            self.state['tgt_fg_tensor'] = None
            self.state['record_view_scores'] = False
            self.state['use_patch_routing'] = False
            _flush_cuda_memory()

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

    ref_imgs = [Image.open('references/1.jpeg')]
    ref_msks = [Image.open('references/1_mask.png')]

    def fidelity_func(s: int, n: int) -> float:
        return float(np.linspace(0.0, 0.5, n)[s])

    def bias_func(s: int, n: int) -> float:
        return float(np.linspace(-12.0, -2.5, n)[s])

    # 1. Precompute reference banks into 3-tier cache (Pinned RAM -> Pageable RAM -> SSD memmap)
    t_ref = time.perf_counter()
    cache_stats = inpainter.set_references(
        reference_image=ref_imgs,
        reference_mask=ref_msks,
        angles_deg=tuple(np.linspace(-15.0, 15.0, 8)),
        scales=tuple(np.linspace(0.70, 1.3, 4)),
        width=512,
        height=512,
        num_inference_steps=35,
        switch_step=1,
        funnel_step=10,
        highres_start_step=4,
        write_stride=2,
        bg_suppress_bias=bias_func,
        isolate_reference=True,
        ref_seed=2,
        max_pinned_gb=2.0,
        min_free_ram_gb=4.0,
        quantize_cache=True,
    )
    print(f'References precomputed in {time.perf_counter() - t_ref:.2f}s | Cache: {cache_stats}')

    # 2. Run fast generation across target images, masks, or seeds
    init_img = Image.open('assets/src.jpg')
    inpaint_msk = Image.open('assets/mask_rect.jpg').filter(ImageFilter.MaxFilter(25))
    attn_msk = inpaint_msk.filter(ImageFilter.GaussianBlur(3))
    blured_mask = inpaint_msk.filter(ImageFilter.MaxFilter(25)).filter(ImageFilter.GaussianBlur(45))

    for target_seed in (2, 7, 19):
        t0 = time.perf_counter()
        output = inpainter.generate(
            init_image=init_img,
            inpaint_mask=blured_mask,
            attn_mask=attn_msk,
            bg_suppress_bias=bias_func,
            top_k_views=15,
            funnel_top_k_views=4,
            funnel_step=10,
            patch_grid_size=2,
            patch_start_step=15,
            num_inference_steps=35,
            switch_step=1,
            guidance_scale=6.5,
            style_fidelity=fidelity_func,
            seed=target_seed,
        )
        print(f'[seed={target_seed}] Generated in {time.perf_counter() - t0:.2f}s')

    output.save('output_fast_multiview_masactrl_sd15.png')
    plt.imshow(output)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
