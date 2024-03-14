from .base import PluginBase
from typing import Tuple

def parse_size(size: str) -> Tuple[int, int]:
    width, height =  size.split('x') if 'x' in size else (size, size)
    return (int(width), int(height))

class PluginSize(PluginBase):

    def setup_args(self, parser):
        parser.add_argument("--size", type=str, default="1024", help="Size 1024x576, 512, 832x1216")

    def setup_pipe(self):
        width, height = parse_size(self.ctx.args.size)
        self.ctx.pipe_opts.width = width
        self.ctx.pipe_opts.height = height

    def pre_pipe(self):
        if 'size' in self.ctx.pipe_opts_otg:
            width, height = parse_size(self.ctx.pipe_opts_otg.pop('size'))
            self.ctx.pipe_opts.width = width
            self.ctx.pipe_opts.height = height
