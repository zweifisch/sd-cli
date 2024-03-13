from argparse import Namespace
import torch
from .base import PluginBase
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image

class PluginImage(PluginBase):

    def setup_args(self, parser):
        parser.add_argument("--image", type=str, help="Image Prompt")
        parser.add_argument("--strength", type=float, default=0.5, help="higher strength -> more creativity")

    def setup_pipeline(self):
        if not self.ctx.args.image:
            return

        self.ctx.pipeline = AutoPipelineForImage2Image
        self.ctx.model = 'stabilityai/stable-diffusion-xl-refiner-1.0'
        self.ctx.pipe_opts.image = load_image(self.ctx.args.image)
        self.ctx.pipe_opts_extra['strength'] = self.ctx.args.strength
