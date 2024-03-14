import os
from argparse import ArgumentParser
from huggingface_hub import hf_hub_download
from diffusers.utils import load_image
from .base import PluginBase
from typing import List

def load_images(locations: List[str]):
    if '.' in os.path.basename(locations[0]):
        return [load_image(x) for x in locations]
    return [load_image(x) for x in sorted([os.path.join(locations[0], basename) for basename in os.listdir(locations[0])])]

class PluginPhotoMaker(PluginBase):

    def setup_args(self, parser: ArgumentParser):
        group = parser.add_argument_group("PhotoMaker")
        group.add_argument("--photo-maker", type=str, nargs='+', help="Reference Image")
        group.add_argument("--photo-maker-weight", type=float, default=1.0, help="PhotoMaker Weight")

    def setup(self):
        if not self.ctx.args.photo_maker:
            return

        from photomaker.pipeline import PhotoMakerStableDiffusionXLPipeline
        self.ctx.pipeline = PhotoMakerStableDiffusionXLPipeline
        self.ctx.pipe_opts_extra['input_id_images'] = load_images(self.ctx.args.photo_maker)

    def setup_pipe(self):
        if not self.ctx.args.photo_maker:
            return

        pipe = self.ctx.pipe
        args = self.ctx.args
        pipe.load_photomaker_adapter(
            os.path.dirname(hf_hub_download(
                repo_id="TencentARC/PhotoMaker",
                filename="photomaker-v1.bin",
                repo_type="model",
            )),
            subfolder="",
            weight_name="photomaker-v1.bin",
            trigger_word="img"
        )
        self.ctx.loras.append(("photomaker", args.photo_maker_weight))
