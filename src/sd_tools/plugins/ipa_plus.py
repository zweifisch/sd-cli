import torch
from argparse import ArgumentParser
from diffusers import DDIMScheduler
from .base import PluginBase
from .utils import load_images


class PluginIPAdaptorPlus(PluginBase):

    def setup_args(self, parser: ArgumentParser):
        group = parser.add_argument_group('IP-Adapter Plus')
        group.add_argument("--ipa-plus", type=str, nargs='+', help="IPAdater Plus")
        group.add_argument("--ipa-plus-face", type=str, nargs='+', help="IPAdater Plus Face")
        group.add_argument("--ipa-plus-scale", type=float, nargs='+', help="IPAdater Plus Scale")

    def setup(self):
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
        pipe.set_ip_adapter_scale(args.ipa_plus_scale)

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
