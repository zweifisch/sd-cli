
from argparse import ArgumentParser, Namespace
from .base import PluginBase

class PluginLora(PluginBase):

    def setup_args(self, parser: ArgumentParser):
        pass

    def setup_pipe(self):
        pipe = self.ctx.pipe
        if len(self.ctx.loras) > 0:
            pipe.set_adapters([x[0] for x in self.ctx.loras], adapter_weights=[x[1] for x in self.ctx.loras])
