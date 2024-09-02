from argparse import Namespace
from .base import PluginBase

class PluginPrompt(PluginBase):

    def setup_args(self, parser):
        parser.add_argument("prompt", type=str, nargs="?", help="Prompt")

    def setup_pipe(self):
        self.ctx.pipe_opts.prompt = self.ctx.args.prompt

    def pre_pipe(self):
        if 'prompt' in self.ctx.pipe_opts_otg:
            self.ctx.pipe_opts.prompt = self.ctx.pipe_opts_otg.pop('prompt').strip()
