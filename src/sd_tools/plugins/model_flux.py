from diffusers import FluxPipeline
from .base import PluginBase

class PluginModelFlux(PluginBase):

    def setup_args(self, parser):
        parser.add_argument("--model", type=str, help="Model")

    def setup(self):
        if self.ctx.args.model:
            self.ctx.model = self.ctx.args.model
        if not self.ctx.model:
            self.ctx.model = "black-forest-labs/FLUX.1-schnell"
        if not self.ctx.pipeline:
            self.ctx.pipeline = FluxPipeline
