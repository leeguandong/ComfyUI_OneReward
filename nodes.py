# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

import torch
import numpy as np
from PIL import Image
import comfy.model_management as mm


class OneRewardModelLoader:
    """加载 OneReward 模型"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (["OneReward", "OneRewardDynamic"],),
            }
        }
    
    RETURN_TYPES = ("ONEREWARD_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load_model"
    CATEGORY = "OneReward"

    def load_model(self, model_type):
        from diffusers import FluxTransformer2DModel
        from .pipeline_flux_fill_with_cfg import FluxFillCFGPipeline
        
        # 根据模型类型选择子文件夹
        if model_type == "OneReward":
            subfolder = "flux.1-fill-dev-OneReward-transformer"
        else:
            subfolder = "flux.1-fill-dev-OneRewardDynamic-transformer"
        
        print(f"[OneReward] Loading {model_type} model...")
        
        # 加载 transformer
        transformer = FluxTransformer2DModel.from_pretrained(
            "bytedance-research/OneReward",
            subfolder=subfolder,
            torch_dtype=torch.bfloat16
        )
        
        # 加载 pipeline
        pipe = FluxFillCFGPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            transformer=transformer,
            torch_dtype=torch.bfloat16
        )
        
        # 移动到设备
        device = mm.get_torch_device()
        pipe = pipe.to(device)
        
        print(f"[OneReward] Model loaded successfully!")
        
        return (pipe,)


class OneRewardSampler:
    """OneReward 采样器"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("ONEREWARD_PIPE",),
                "image": ("IMAGE",),
                "mask": ("IMAGE",),  # ComfyUI mask 作为 IMAGE 输入
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "nsfw"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 30.0,
                    "step": 0.1
                }),
                "true_cfg": ("FLOAT", {
                    "default": 4.0,
                    "min": 1.0,
                    "max": 15.0,
                    "step": 0.1
                }),
                "num_inference_steps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 200
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sample"
    CATEGORY = "OneReward"

    def sample(self, pipe, image, mask, prompt, negative_prompt, guidance_scale, true_cfg, num_inference_steps, seed):
        # 转换 ComfyUI image format (B,H,W,C) to PIL
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        
        # 转换 mask to PIL
        mask_np = (mask[0].cpu().numpy() * 255).astype(np.uint8)
        pil_mask = Image.fromarray(mask_np).convert('L')
        
        # 创建生成器
        generator = torch.Generator("cpu").manual_seed(seed)
        
        print(f"[OneReward] Sampling...")
        print(f"[OneReward] Prompt: {prompt[:100]}...")
        print(f"[OneReward] Steps: {num_inference_steps}, CFG: {true_cfg}")
        
        # 执行推理（完全按照原始代码的参数）
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pil_image,
            mask_image=pil_mask,
            height=pil_image.height,
            width=pil_image.width,
            guidance_scale=guidance_scale,
            true_cfg=true_cfg,
            num_inference_steps=num_inference_steps,
            generator=generator
        ).images[0]
        
        print(f"[OneReward] Done!")
        
        # 转换回 ComfyUI 格式
        result_np = np.array(result).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np)[None,]
        
        return (result_tensor,)


NODE_CLASS_MAPPINGS = {
    "OneRewardModelLoader": OneRewardModelLoader,
    "OneRewardSampler": OneRewardSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OneRewardModelLoader": "OneReward Model Loader",
    "OneRewardSampler": "OneReward Sampler",
}
