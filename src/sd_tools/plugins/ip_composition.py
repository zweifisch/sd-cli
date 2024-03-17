import torch
from argparse import ArgumentParser
from diffusers import DDIMScheduler
from .base import PluginBase
from .utils import load_images


class PluginIPCompositionAdapter(PluginBase):

    def setup_args(self, parser: ArgumentParser):
        group = parser.add_argument_group('IP Composition Adapter')
        group.add_argument("--ip-composition", type=str, nargs='+', help="Reference image")
        group.add_argument("--ip-composition-scale", type=float, default=0.6, help="IPAdapter scale")

    def setup(self):
        if not self.ctx.args.ip_composition:
            return

        from transformers import CLIPVisionModelWithProjection
        self.ctx.pipeline_opts_extra['image_encoder'] = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=torch.float16,
            local_files_only=self.ctx.offline,
            resume_download=not self.ctx.offline,
        )

    def setup_pipe(self):
        if not self.ctx.args.ip_composition:
            return

        pipe = self.ctx.pipe
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_ip_adapter_scale(self.ctx.args.ip_composition_scale)
        self.ctx.pipe_opts_extra['ip_adapter_image'] = load_images(self.ctx.args.ip_composition)

        pipe.load_ip_adapter(
            "ostris/ip-composition-adapter",
            subfolder='',
            weight_name="ip_plus_composition_sd15.safetensors",
        )
