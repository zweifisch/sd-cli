import os
import torch
from argparse import ArgumentParser
from diffusers import DDIMScheduler
from .base import PluginBase
from diffusers.utils import load_image
from typing import List

def load_images(locations: List[str]):
    if locations is None:
        return []
    if '.' in os.path.basename(locations[0]):
        return [load_image(x) for x in locations]
    return [load_image(x) for x in sorted([os.path.join(locations[0], basename) for basename in os.listdir(locations[0])])]

class PluginIPAdaptorPlus(PluginBase):

    def setup_args(self, parser: ArgumentParser):
        parser.add_argument("--ipa-plus", type=str, nargs='+', help="IPAdator Plus")
        parser.add_argument("--ipa-plus-face", type=str, nargs='+', help="IPAdator Plus Face")

    def setup_pipeline(self):
        if not self.ctx.args.ipa_plus_face and not self.ctx.args.ipa_plus:
            return

        from transformers import CLIPVisionModelWithProjection
        self.ctx.pipeline_opts_extra['image_encoder'] = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=torch.float16,
            resume_download=not self.ctx.offline,
        )

    def setup_pipe(self):
        args = self.ctx.args
        if not args.ipa_plus_face and not args.ipa_plus:
            return

        pipe = self.ctx.pipe
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_ip_adapter_scale(args.ipa_scale)

        pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name=[
                x[0] for x in [
                    ("ip-adapter-plus_sdxl_vit-h.safetensors", args.ipa_plus),
                    ("ip-adapter-plus-face_sdxl_vit-h.safetensors", args.ipa_plus_face)
                ] if x[1]
            ]
        )
        self.ctx.pipe_opts_extra['ip_adapter_image'] = [
            x[0] for x in [
                (load_images(args.ipa_plus), args.ipa_plus),
                (load_images(args.ipa_plus_face), args.ipa_plus_face)
            ] if x[1]
        ]
