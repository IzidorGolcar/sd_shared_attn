"""Stable Diffusion multi-reference inpainting pipeline."""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
from tqdm.auto import tqdm

from .attention_processors import BackgroundSuppressedAttnProcessor, MultiViewMasaCtrlAttnProcessor
from .reference_cache import TieredReferenceCache
from .reference_utils import _build_augmented_views, _flush_cuda_memory, _default_bg_suppress_bias, _default_style_fidelity

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
        augment_pbar = tqdm(
            zip(ref_images, ref_masks),
            total=len(ref_images),
            desc='References: augment views',
            unit='image',
        )
        for ref_img, ref_msk in augment_pbar:
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
            augment_pbar.set_postfix(
                views=len(aug_images),
                angle_count=len(angles_deg),
                scale_count=len(scales),
            )
        augment_pbar.close()
        n_views = len(aug_images)

        ref_fg_tensors = torch.from_numpy(
            np.stack([(np.array(m, dtype=np.uint8) > 127).astype(np.float32) for m in aug_masks], axis=0)
        )[:, None].to(device=self.device, dtype=self.dtype)  # [N, 1, H, W]

        # Encode in chunks directly from PIL -> tensor so full N-image pixel tensor is never allocated on GPU
        latent_chunks = []
        latent_starts = range(0, n_views, self.batch_chunk_size)
        latent_pbar = tqdm(
            latent_starts,
            total=(n_views + self.batch_chunk_size - 1) // self.batch_chunk_size,
            desc='References: encode VAE latents',
            unit='chunk',
        )
        for start in latent_pbar:
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
            latent_pbar.set_postfix(view=f'{end}/{n_views}', size=f'{end - start}')
        latent_pbar.close()

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

        write_pbar = tqdm(
            write_steps,
            desc='References: build UNet banks',
            unit='step',
        )
        try:
            for step_idx in write_pbar:
                bias_val = (
                    float(bg_suppress_bias(step_idx, len(timesteps)))
                    if callable(bg_suppress_bias)
                    else float(bg_suppress_bias)
                )
                if bias_val > 0.0:
                    raise ValueError('bg_suppress_bias must be <= 0.0')
                self.state['bg_suppress_bias'] = bias_val
                self.state['bank_chunks'].clear()
                write_pbar.set_postfix(
                    step=f'{step_idx}/{len(timesteps)}',
                    timestep=f'{float(timesteps[step_idx]):.1f}',
                    bg_bias=f'{bias_val:.2f}',
                    views=n_views,
                )

                self.state['write_skip_prefixes'] = (
                    ('up_blocks.3',) if step_idx < int(highres_start_step) else ()
                )

                t = timesteps[step_idx]
                self.state['mode'] = 'write'
                view_starts = range(0, n_views, self.batch_chunk_size)
                chunk_pbar = tqdm(
                    view_starts,
                    total=(n_views + self.batch_chunk_size - 1) // self.batch_chunk_size,
                    desc=f'References: step {step_idx}',
                    unit='view-chunk',
                    leave=False,
                )
                for start in chunk_pbar:
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
                    chunk_pbar.set_postfix(
                        views=f'{end}/{n_views}',
                        chunk=f'{b_chunk}',
                    )
                chunk_pbar.close()

                self.state['write_fg_masks'] = None
                self.state['write_slice'] = None

                bank_layers = tqdm(
                    self.state['bank_chunks'].items(),
                    total=len(self.state['bank_chunks']),
                    desc=f'References: cache step {step_idx}',
                    unit='layer',
                    leave=False,
                )
                for layer_name, view_list in bank_layers:
                    store.put_layer(step_idx, layer_name, view_list)
                bank_layers.close()
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
            write_pbar.close()
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
