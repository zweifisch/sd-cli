from argparse import Namespace
import os
from .base import PluginBase
from datetime import datetime

class PluginOutput(PluginBase):

    def setup_args(self, parser):
        parser.add_argument("--output", type=str, default="output", help="output dir")

    def setup_pipeline(self):
        if self.ctx.args.output:
            os.makedirs(self.ctx.args.output, exist_ok=True)

    def post_pipe(self, result):
        time = datetime.now().timestamp()
        result.images[0].save(os.path.join(self.ctx.args.output, f"{self.ctx.seed}-{time}.webp"))
