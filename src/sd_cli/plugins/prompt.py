from argparse import Namespace
from .base import PluginBase

class PluginPrompt(PluginBase):

    def setup_args(self, parser):
        parser.add_argument("--prompt", type=str, default="", help="prompt")
        parser.add_argument("--negative-prompt", type=str, default="worst quality, low quality, blurry, inappropriate pose, deformed, extra limbs, bad posture, bad makeup, watermark, signature, caption, bad eye, open mouth", help="negative prompt")

    def setup_pipe(self):
        self.ctx.pipe_opts.prompt = self.ctx.args.prompt
        self.ctx.pipe_opts.negative_prompt = self.ctx.args.negative_prompt
