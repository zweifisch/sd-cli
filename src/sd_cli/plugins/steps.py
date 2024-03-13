from argparse import Namespace
from .base import PluginBase

class PluginSteps(PluginBase):

    def setup_args(self, parser):
        parser.add_argument("--steps", type=int, default=4, help="steps")

    def setup_pipe(self):
        self.ctx.pipe_opts.num_inference_steps = self.ctx.args.steps
