"""Tiered host-memory and SSD storage for reference token banks."""
import gc
import os
import shutil
import tempfile
import weakref
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from .reference_utils import _get_available_ram_bytes

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


