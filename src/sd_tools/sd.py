import argparse
from argparse import RawTextHelpFormatter

from .plugins.debug import PluginDebug
from .plugins.base import Context, PipeOptions, PipelineOptions
from .plugins.cfg import PluginCFG
from .plugins.size import PluginSize
from .plugins.prompt import PluginPrompt
from .plugins.model import PluginModel
from .plugins.pipe import PluginPipe
from .plugins.inpainting import PluginInpainting
from .plugins.steps import PluginSteps
from .plugins.output import PluginOutput
from .plugins.lora import PluginLora
from .plugins.seed import PluginSeed
from .plugins.scheduler import PluginScheduler
from .plugins.device import PluginDevice
from .plugins.lcm import PluginLCM
from .plugins.offline import PluginOffline
from .plugins.http import PluginHTTP
from .plugins.run import PluginRun

def main():
    parser = argparse.ArgumentParser(
        description="Stable Diffusion",
        usage="",
        formatter_class=RawTextHelpFormatter,
        epilog="""
Examples:

    %(prog)s 'locomotive comming'

    %(prog)s 'locomotive comming' --steps 4 --model Lykon/dreamshaper-8 --lcm 1 --cfg 2

"""
    )

    ctx = Context(arch='SD')
    plugins = [
        PluginDebug(ctx),
        PluginPipe(ctx),
        PluginOffline(ctx),
        PluginInpainting(ctx),
        PluginModel(ctx),
        PluginDevice(ctx),
        PluginSize(ctx),
        PluginCFG(ctx),
        PluginPrompt(ctx),
        PluginSteps(ctx),
        PluginSeed(ctx),
        PluginLCM(ctx),
        PluginOutput(ctx),
        # PluginCanny(ctx),
        # PluginPose(ctx),
        # PluginDepth(ctx),
        # PluginIPAdaptor(ctx),
        # PluginIPAdaptorPlus(ctx),
        # PluginIPAdaptorFaceID(ctx),
        # PluginIPAdaptorFaceIDPlus(ctx),
        # PluginImage(ctx),
        PluginScheduler(ctx),
        PluginLora(ctx),
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
