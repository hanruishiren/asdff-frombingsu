from __future__ import annotations

from functools import cached_property

from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionControlNetPipeline, 
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
)

#【增加】导入FromSingleFileMixin
from diffusers.loaders import FromSingleFileMixin
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import from_ckpt

from pathlib import Path

from asdff.base import AdPipelineBase

#【增加】继承FromSingleFileMixin
class AdPipeline(AdPipelineBase, StableDiffusionPipeline, FromSingleFileMixin):

    #【增加】实现from_single_file方法
    @classmethod
    def from_single_file(cls, pretrained_model_link_or_path, **kwargs):
        cache_dir = kwargs.pop("cache_dir", "diffusers_cache")
        
        # 删除url前缀
        for prefix in ["https://huggingface.co/"]:
            if pretrained_model_link_or_path.startswith(prefix):
                pretrained_model_link_or_path = pretrained_model_link_or_path[len(prefix):]
                
        # 如果提供的是url,下载文件        
        ckpt_path = Path(pretrained_model_link_or_path)
        if not ckpt_path.is_file():
            # 通过hf_hub_download API下载
            pretrained_model_link_or_path = hf_hub_download(
                repo_id=ckpt_path.parts[0], 
                file_path=ckpt_path.parts[1:],
                cache_dir=cache_dir
            )
        
        # 调用from_ckpt函数加载
        pipe = from_ckpt(
            pretrained_model_link_or_path,
            pipeline_class=cls,
            **kwargs
        )
        
        return pipe

    @cached_property
    def inpaint_pipeline(self):
        return StableDiffusionInpaintPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.scheduler,
            safety_checker=self.safety_checker,
            feature_extractor=self.feature_extractor,
            requires_safety_checker=self.config.requires_safety_checker,
        )

    @property
    def txt2img_class(self):
        return StableDiffusionPipeline

#【增加】继承FromSingleFileMixin
class AdCnPipeline(AdPipelineBase, StableDiffusionControlNetPipeline, FromSingleFileMixin):

    #【增加】实现from_single_file方法
    @classmethod
    def from_single_file(cls, pretrained_model_link_or_path, **kwargs):
        cache_dir = kwargs.pop("cache_dir", "diffusers_cache")
        
        # 删除url前缀
        for prefix in ["https://huggingface.co/"]:
            if pretrained_model_link_or_path.startswith(prefix):
                pretrained_model_link_or_path = pretrained_model_link_or_path[len(prefix):]
                
        # 如果提供的是url,下载文件        
        ckpt_path = Path(pretrained_model_link_or_path)
        if not ckpt_path.is_file():
            # 通过hf_hub_download API下载
            pretrained_model_link_or_path = hf_hub_download(
                repo_id=ckpt_path.parts[0], 
                file_path=ckpt_path.parts[1:],
                cache_dir=cache_dir
            )
        
        # 调用from_ckpt函数加载
        pipe = from_ckpt(
            pretrained_model_link_or_path,
            pipeline_class=cls,
            **kwargs
        )
        
        return pipe

    @cached_property
    def inpaint_pipeline(self):
        return StableDiffusionControlNetInpaintPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            controlnet=self.controlnet,
            scheduler=self.scheduler,
            safety_checker=self.safety_checker,
            feature_extractor=self.feature_extractor,
            requires_safety_checker=self.config.requires_safety_checker,
        )

    @property
    def txt2img_class(self):
        return StableDiffusionControlNetPipeline
