import argparse
from argparse import RawTextHelpFormatter

from sd_tools.plugins.safetensor import PluginSafetensor

from .plugins.debug import PluginDebug
from .plugins.base import Context, PipeOptions, PipelineOptions
from .plugins.cfg import PluginCFG
from .plugins.size import PluginSize
from .plugins.prompt import PluginPrompt
from .plugins.auraflow import PluginAuraflow
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
        description="Aura Flow",
        usage="",
        formatter_class=RawTextHelpFormatter,
        epilog="""
Examples:

    %(prog)s 'portrait of a man' --steps 50 --size 1024 --cfg 3.5 --no-fp16

"""
    )

    ctx = Context(arch='SD3')
    plugins = [
        PluginDebug(ctx),
        PluginPipe(ctx),
        PluginOffline(ctx),
        PluginSafetensor(ctx),
        PluginInpainting(ctx),
        PluginAuraflow(ctx),
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
