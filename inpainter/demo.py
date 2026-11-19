"""Command-line demonstration for the multi-reference inpainter."""
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

from .inpainter import FastMultiViewMasaCtrlInpainter


def run_demo() -> None:
    import time
    import matplotlib.pyplot as plt
    
    inpainter = FastMultiViewMasaCtrlInpainter(
        inpaint_model_id='sd-legacy/stable-diffusion-inpainting',
        device='cuda',
        hook_blocks=('up_blocks.2', 'up_blocks.3'),
        batch_chunk_size=8,
    )
    
    ref_imgs = [Image.open('../references/1.jpeg')]
    ref_msks = [Image.open('../references/1_mask.png')]
    
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
    init_img = Image.open('../assets/src.jpg')
    inpaint_msk = Image.open('../assets/mask_rect.jpg').filter(ImageFilter.MaxFilter(25))
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
