from dataclasses import asdict
import os
import io
import sys
import torch
import argparse
from argparse import Namespace
import time
import hashlib
import random

from .http import serve

from .plugins.debug import PluginDebug
from .plugins.base import Context, PipeOptions, PipelineOptions
from .plugins.lightning import PluginLightning
from .plugins.photo_maker import PluginPhotoMaker
from .plugins.cfg import PluginCFG
from .plugins.size import PluginSize
from .plugins.prompt import PluginPrompt
from .plugins.sdxl import PluginSDXL
from .plugins.tcd import PluginTCD
from .plugins.lcm import PluginLCM
from .plugins.steps import PluginSteps
from .plugins.output import PluginOutput
from .plugins.lora import PluginLora
from .plugins.seed import PluginSeed
from .plugins.scheduler import PluginScheduler
from .plugins.res_adapter import PluginResAdaptor
from .plugins.image import PluginImage
from .plugins.device import PluginDevice
from .plugins.ipa import PluginIPAdaptor
from .plugins.ipa_plus import PluginIPAdaptorPlus
from .plugins.ipa_faceid import PluginIPAdaptorFaceID
from .plugins.ipa_faceid_plus import PluginIPAdaptorFaceIDPlus
from .plugins.instantid import PluginInstantID
from .plugins.canny import PluginCanny
from .plugins.pose import PluginPose
from .plugins.depth import PluginDepth

def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion cli")
    parser.add_argument('--listen', type=str)
    parser.add_argument('--count', default=1, type=int)

    ctx = Context()
    plugins = [
        PluginDebug(ctx),
        PluginDevice(ctx),
        PluginSize(ctx),
        PluginCFG(ctx),
        PluginPrompt(ctx),
        PluginSteps(ctx),
        PluginSeed(ctx),
        PluginSDXL(ctx),
        PluginTCD(ctx),
        PluginLCM(ctx),
        PluginOutput(ctx),
        PluginLightning(ctx),
        PluginPhotoMaker(ctx),
        PluginCanny(ctx),
        PluginPose(ctx),
        PluginDepth(ctx),
        PluginIPAdaptor(ctx),
        PluginIPAdaptorPlus(ctx),
        PluginIPAdaptorFaceID(ctx),
        PluginIPAdaptorFaceIDPlus(ctx),
        PluginInstantID(ctx),
        PluginResAdaptor(ctx),
        PluginImage(ctx),
        PluginLora(ctx),
        PluginScheduler(ctx),
    ]

    for plugin in plugins:
        plugin.setup_args(parser)

    ctx.args = parser.parse_args()

    for plugin in plugins:
        plugin.setup_pipeline()

    ctx.pipe =  ctx.pipeline.from_pretrained(ctx.model, **{**asdict(ctx.pipeline_opts), **ctx.pipeline_opts_extra}).to(ctx.device)

    for plugin in plugins:
        plugin.setup_pipe()

    if ctx.args.listen:
        serve(ctx.args.listen, lambda payload: ctx.pipe(**{**asdict(ctx.pipe_opts), **ctx.pipe_opts_extra, **payload}))
        return



    for no in range(ctx.args.count):

        for plugin in plugins:
            plugin.pre_pipe()

        result = ctx.pipe(**remove_none(asdict(ctx.pipe_opts)), **ctx.pipe_opts_extra)

        for plugin in plugins:
            plugin.post_pipe(result)

def remove_none(input_dict):
    return {k: v for k, v in input_dict.items() if v is not None}
