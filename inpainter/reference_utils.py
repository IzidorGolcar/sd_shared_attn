"""Image augmentation, memory, and token extraction helpers for reference banks."""
import gc
import math
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

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


