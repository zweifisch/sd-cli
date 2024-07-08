import argparse
from argparse import RawTextHelpFormatter

from sd_tools.plugins.safetensor import PluginSafetensor

from .plugins.debug import PluginDebug
from .plugins.base import Context, PipeOptions, PipelineOptions
from .plugins.cfg import PluginCFG
from .plugins.size import PluginSize
from .plugins.prompt import PluginPrompt
from .plugins.model3 import PluginModel3
from .plugins.pipe import PluginPipe
from .plugins.inpainting import PluginInpainting
from .plugins.steps import PluginSteps
from .plugins.output import PluginOutput
from .plugins.seed import PluginSeed
from .plugins.output import PluginOutput
from .plugins.device import PluginDevice
from .plugins.offline import PluginOffline
from .plugins.http import PluginHTTP
from .plugins.run import PluginRun

def main():
    parser = argparse.ArgumentParser(
        description="Stable Diffusion 3",
        usage="",
        formatter_class=RawTextHelpFormatter,
        epilog="""
Examples:

    %(prog)s 'locomotive comming, masterpiece'

    %(prog)s 'portrait of a man' --steps 6 --size 576x1024 --cfg 1.5

"""
    )

    ctx = Context(arch='SD3')
    plugins = [
        PluginDebug(ctx),
        PluginPipe(ctx),
        PluginOffline(ctx),
        PluginSafetensor(ctx),
        PluginInpainting(ctx),
        PluginModel3(ctx),
        PluginDevice(ctx),
        PluginSize(ctx),
        PluginCFG(ctx),
        PluginPrompt(ctx),
        PluginSteps(ctx),
        PluginSeed(ctx),
        PluginOutput(ctx),
        PluginHTTP(ctx),
        PluginRun(ctx),
    ]

    ctx.plugins = plugins

    for plugin in plugins:
        plugin.setup_args(parser)

    ctx.args = parser.parse_args()

    for plugin in plugins:
        plugin.setup()

    for plugin in plugins:
        plugin.setup_pipe()
