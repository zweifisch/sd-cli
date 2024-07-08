from diffusers import StableDiffusion3Pipeline
from .base import PluginBase

class PluginModel3(PluginBase):

    def setup_args(self, parser):
        parser.add_argument("--model", type=str, help="Model")

    def setup(self):
        if self.ctx.args.model:
            self.ctx.model = self.ctx.args.model
        if not self.ctx.model:
            self.ctx.model = "stabilityai/stable-diffusion-3-medium-diffusers"
        if not self.ctx.pipeline:
            self.ctx.pipeline = StableDiffusion3Pipeline
