import torch
from argparse import ArgumentParser
from diffusers.utils import load_image
from .base import PluginBase
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from PIL import Image
from transformers import pipeline

def resize_image(image, width, height):
    width0, height0 = image.size

    fit_width = width0 / height0 > width / height

    new_width = width if fit_width else int(width0 * height / height0)
    new_height = height if not fit_width else int(height0 * width / width0)

    resized = image.resize((new_width // 8 * 8, new_height // 8 * 8))
    return resized

    new_image = Image.new("RGB", (width, height), (0, 0, 0))
    new_image.paste(resized, ((width - new_width) // 2, (height - new_height) // 2))

    return new_image

class PluginDepth(PluginBase):

    def setup_args(self, parser: ArgumentParser):
        parser.add_argument('--depth', type=str)

    def setup_pipeline(self):
        if not self.ctx.args.depth:
            return

        pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
        self.depth = pipe(load_image(self.ctx.args.depth))["depth"]

        self.ctx.pipeline_opts_extra['controlnet'] = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0-small",
            torch_dtype=torch.float16,
            variant='fp16',
            use_safetensors=True,
            resume_download=True
        )
        self.ctx.pipeline = StableDiffusionXLControlNetPipeline

    def setup_pipe(self):
        if not self.ctx.args.depth:
            return

        resized = resize_image(self.depth, self.ctx.pipe_opts.width, self.ctx.pipe_opts.height)
        self.ctx.pipe_opts.image = resized
        self.ctx.pipe_opts.width = resized.width
        self.ctx.pipe_opts.height = resized.height
