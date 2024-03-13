from argparse import Namespace
import torch
from .base import PluginBase
from diffusers import StableDiffusionXLPipeline

class PluginSDXL(PluginBase):

    def setup_args(self, parser):
        parser.add_argument("--model", type=str, default="stabilityai/sdxl-turbo", help="model")
        parser.add_argument("--no-fp16", action="store_true", default=False, help="no fp16 weights")
        parser.add_argument("--no-safetensor", action="store_true", default=False, help="no safetensor")

    def setup_pipeline(self):
        self.ctx.model = self.ctx.args.model
        self.ctx.pipeline = StableDiffusionXLPipeline
        self.ctx.pipeline_opts.torch_dtype = torch.float16
        if self.ctx.args.no_fp16:
            self.ctx.pipeline_opts.variant = None
        if self.ctx.args.no_safetensor:
            self.ctx.pipeline_opts.use_safetensors = False
