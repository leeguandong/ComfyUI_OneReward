# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch

from diffusers.pipelines.flux.pipeline_flux_fill import (
    FluxFillPipeline,
    FluxPipelineOutput,
    calculate_shift,
    retrieve_timesteps,
)


class FluxFillCFGPipeline(FluxFillPipeline):
    """
    Wrapper of `FluxFillPipeline` that adds *classifier-free guidance* (CFG) for the Fill pipeline.

    Key idea ("true CFG"):
      - In each denoising step, run the transformer on a batch that is the
        concatenation of [unconditional, conditional] inputs, then mix the two
        predictions:  pred = pred_neg + true_cfg * (pred_pos - pred_neg).

    New args:
      - negative_prompt:     str | List[str] for unconditional branch (CLIP/T5)
      - negative_prompt_2:   str | List[str] for the second encoder (T5) if you
                              want it different from `negative_prompt`.
      - true_cfg:           float (default 1.0). If <= 1.0, behaves like upstream.
    """

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: Optional[torch.FloatTensor] = None,
        mask_image: Optional[torch.FloatTensor] = None,
        masked_image_latents: Optional[torch.FloatTensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 1.0,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 30.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        # --- CFG additions ---
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg: float = 1.0,
    ):
        # ---------- 0) Defaults / basic checks ----------
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # Validate base inputs via upstream checker
        self.check_inputs(
            prompt,
            prompt_2,
            strength,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
            image=image,
            mask_image=mask_image,
            masked_image_latents=masked_image_latents,
        )

        # Internal flags / state
        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # Preprocess input image (BCHW, float32 in [0,1])
        init_image = self.image_processor.preprocess(image, height=height, width=width)
        init_image = init_image.to(dtype=torch.float32)

        # Batch size logic matches upstream pipeline
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # ---------- 1) Encode positive (and, if needed, negative) prompts ----------
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )

        # Positive
        pos_prompt_embeds, pos_pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        use_cfg = true_cfg is not None and true_cfg > 1.0

        # Negative (unconditional). If not provided and CFG is requested, fall back to empty prompt
        if use_cfg:
            if negative_prompt is None:
                negative_prompt = [""] * (batch_size)
            neg_prompt_embeds, neg_pooled_prompt_embeds, _ = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )
        else:
            neg_prompt_embeds = None
            neg_pooled_prompt_embeds = None

        # ---------- 2) Timesteps (same as upstream) ----------
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = (int(height) // self.vae_scale_factor // 2) * (int(width) // self.vae_scale_factor // 2)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        if num_inference_steps < 1:
            raise ValueError(
                f"After adjusting the num_inference_steps by strength={strength}, steps is {num_inference_steps} (<1)."
            )
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # ---------- 3) Prepare latents and the (packed) masked image latents ----------
        num_channels_latents = self.vae.config.latent_channels
        latents, latent_image_ids = self.prepare_latents(
            init_image,
            latent_timestep,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            pos_prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        if masked_image_latents is not None:
            masked_image_latents = masked_image_latents.to(latents.device)
        else:
            mask_image_tensor = self.mask_processor.preprocess(mask_image, height=height, width=width)
            masked_image = init_image * (1 - mask_image_tensor)
            masked_image = masked_image.to(device=device, dtype=pos_prompt_embeds.dtype)

            h_img, w_img = init_image.shape[-2:]
            mask, masked_image_latents = self.prepare_mask_latents(
                mask_image_tensor,
                masked_image,
                batch_size,
                num_channels_latents,
                num_images_per_prompt,
                h_img,
                w_img,
                pos_prompt_embeds.dtype,
                device,
                generator,
            )
            masked_image_latents = torch.cat((masked_image_latents, mask), dim=-1)

        # ---------- 4) Guidance embeds (the model's native guidance, not CFG) ----------
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # ---------- 5) Denoising loop with optional true-CFG ----------
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # timestep per-sample (matches upstream)
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                if use_cfg:
                    # Build a 2x batch: [uncond, cond]
                    latents_in = torch.cat([latents, latents], dim=0)
                    masked_in = torch.cat([masked_image_latents, masked_image_latents], dim=0)
                    pooled_in = torch.cat([neg_pooled_prompt_embeds, pos_pooled_prompt_embeds], dim=0)
                    prompt_in = torch.cat([neg_prompt_embeds, pos_prompt_embeds], dim=0)
                    guidance_in = guidance.repeat(2) if guidance is not None else None
                    timestep_in = t.expand(latents_in.shape[0]).to(latents.dtype)

                    noise_pred_all = self.transformer(
                        hidden_states=torch.cat((latents_in, masked_in), dim=2),
                        timestep=timestep_in / 1000,
                        guidance=guidance_in,
                        pooled_projections=pooled_in,
                        encoder_hidden_states=prompt_in,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred_neg, noise_pred_pos = noise_pred_all.chunk(2, dim=0)
                    noise_pred = noise_pred_neg + true_cfg * (noise_pred_pos - noise_pred_neg)
                else:
                    # Upstream single-branch path
                    noise_pred = self.transformer(
                        hidden_states=torch.cat((latents, masked_image_latents), dim=2),
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pos_pooled_prompt_embeds,
                        encoder_hidden_states=pos_prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]

                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        if k == "prompt_embeds":
                            callback_kwargs[k] = pos_prompt_embeds
                        else:
                            callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    pos_prompt_embeds = callback_outputs.pop("prompt_embeds", pos_prompt_embeds)

                # Progress
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # ---------- 6) Decode / Postprocess ----------
        if output_type == "latent":
            image = latents
        else:
            latents_unpacked = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents_unpacked = (latents_unpacked / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents_unpacked, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)
        return FluxPipelineOutput(images=image)
