"""Compatibility facade for the multi-reference inpainting implementation.

The implementation is split by responsibility, while these imports preserve
the original public module surface.
"""

from .attention_processors import (
    BackgroundSuppressedAttnProcessor,
    MultiViewMasaCtrlAttnProcessor,
    _build_unit_isolation_weights,
    _get_isolation_bias,
    _isolated_self_attention,
)
from .demo import run_demo
from .inpainter import FastMultiViewMasaCtrlInpainter
from .reference_cache import TieredReferenceCache
from .reference_utils import (
    _build_augmented_views,
    _default_bg_suppress_bias,
    _default_style_fidelity,
    _flush_cuda_memory,
    _get_available_ram_bytes,
    _mask_centroid,
    extract_foreground_tokens_ragged,
)

__all__ = [
    "BackgroundSuppressedAttnProcessor",
    "FastMultiViewMasaCtrlInpainter",
    "MultiViewMasaCtrlAttnProcessor",
    "TieredReferenceCache",
    "extract_foreground_tokens_ragged",
]


if __name__ == "__main__":
    run_demo()
