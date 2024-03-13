import os
import torch
from argparse import ArgumentParser
from diffusers import DDIMScheduler
from .base import PluginBase
from diffusers.utils import load_image
from typing import List

def load_images(locations: List[str]):
    if '.' in os.path.basename(locations[0]):
        return [load_image(x) for x in locations]
    return [load_image(x) for x in sorted([os.path.join(locations[0], basename) for basename in os.listdir(locations[0])])]

class PluginIPAdaptor(PluginBase):

    def setup_args(self, parser: ArgumentParser):
        parser.add_argument("--ipa", type=str, nargs='+', help="IP-Adaptor")
        parser.add_argument("--ipa-scale", type=float, nargs='+', help="IP-Adaptor Scale")

    def setup_pipe(self):
        args = self.ctx.args
        if not args.ipa:
            return

        pipe = self.ctx.pipe
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin", resume_download=not self.ctx.offline)
        pipe.set_ip_adapter_scale(args.ipa_scale[0])
        self.ctx.pipe_opts_extra['ip_adapter_image'] = load_image(args.ipa[0])
