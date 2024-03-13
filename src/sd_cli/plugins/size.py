from .base import PluginBase

class PluginSize(PluginBase):

    def setup_args(self, parser):
        parser.add_argument("--size", type=str, default="1024", help="size")

    def setup_pipe(self):
        size = self.ctx.args.size
        width, height =  size.split('x') if 'x' in size else [size, size]
        self.ctx.pipe_opts.width = int(width)
        self.ctx.pipe_opts.height = int(height)
