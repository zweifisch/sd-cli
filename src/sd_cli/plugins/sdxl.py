from dataclasses import asdict
import torch
from .base import PluginBase
from diffusers import StableDiffusionXLPipeline

class PluginSDXL(PluginBase):

    def setup_args(self, parser):
        parser.add_argument("--model", type=str, default="stabilityai/sdxl-turbo", help="Model")
        parser.add_argument("--no-fp16", action="store_true", default=False, help="No fp16 weights")
        parser.add_argument("--no-safetensor", action="store_true", default=False, help="No safetensor")

    def setup(self):
        self.ctx.model = self.ctx.args.model

        self.ctx.pipeline = StableDiffusionXLPipeline
        self.ctx.pipeline_opts.torch_dtype = torch.float16
        if self.ctx.args.no_fp16:
            self.ctx.pipeline_opts.variant = None
        if self.ctx.args.no_safetensor:
            self.ctx.pipeline_opts.use_safetensors = False

    def setup_pipe(self):
        if self.ctx.model.endswith(".safetensors"):
            self.ctx.pipe = self.ctx.pipeline.from_single_file(
                self.ctx.model,
                **{**asdict(self.ctx.pipeline_opts), **self.ctx.pipeline_opts_extra}
            ).to(self.ctx.device)
        else:
            self.ctx.pipe = self.ctx.pipeline.from_pretrained(
                self.ctx.model,
                **{**asdict(self.ctx.pipeline_opts), **self.ctx.pipeline_opts_extra}
            ).to(self.ctx.device)
