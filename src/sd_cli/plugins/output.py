from argparse import Namespace
import os
from .base import PluginBase
from datetime import datetime

class PluginOutput(PluginBase):

    def setup_args(self, parser):
        parser.add_argument("--output", type=str, default="output", help="Output Directory")
        parser.add_argument("--ext", type=str, default="webp", help="Output File Extension")

    def setup(self):
        if self.ctx.args.output:
            os.makedirs(self.ctx.args.output, exist_ok=True)

    def post_pipe(self, result):
        time = datetime.now().timestamp()
        result.images[0].save(os.path.join(self.ctx.args.output, f"{self.ctx.seed}-{time}.{self.ctx.args.ext}"))
