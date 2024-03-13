from .base import PluginBase

class PluginDebug(PluginBase):

    def setup_args(self, parser):
        parser.add_argument("--debug", "-d", type=bool, default=False)

    def setup_pipeline(self):
        self.ctx.debug = self.ctx.args.debug
