from argparse import Namespace
from huggingface_hub import hf_hub_download
import os
import torch
from .base import PluginBase
from diffusers.models import ControlNetModel
from diffusers.utils import load_image
from .utils import resize_image

def hf_download(fullname, offline=False):
    ns, project, filename = fullname.split('/', 2)
    return hf_hub_download(repo_id=f"{ns}/{project}", filename=filename, local_files_only=offline, resume_download=not offline)

class PluginInstantID(PluginBase):

    def setup_args(self, parser):
        parser.add_argument("--instantid", type=str, help="Instant ID")

    def setup_pipeline(self):
        if not self.ctx.args.instantid:
            return

        from .pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline
        ctrlnet_path = hf_download("InstantX/InstantID/ControlNetModel/config.json", offline=self.ctx.offline)
        hf_download("InstantX/InstantID/ControlNetModel/diffusion_pytorch_model.safetensors", offline=self.ctx.offline)

        self.ctx.pipeline_opts_extra['controlnet'] = [ControlNetModel.from_pretrained(
            os.path.dirname(ctrlnet_path),
            torch_dtype=torch.float32
        )]
        self.ctx.pipeline_opts.torch_dtype = torch.float32
        self.ctx.pipeline = StableDiffusionXLInstantIDPipeline

    def setup_pipe(self):
        if not self.ctx.args.instantid:
            return

        from insightface.app import FaceAnalysis
        import cv2
        import numpy as np
        from .pipeline_stable_diffusion_xl_instantid import draw_kps

        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))

        face_image = load_image(self.ctx.args.instantid)
        face_image = resize_image(face_image, self.ctx.pipe_opts.width, self.ctx.pipe_opts.height)
        face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1]  # only use the maximum face
        self.ctx.pipe_opts_extra['image_embeds'] = face_info['embedding']

        pose_image = draw_kps(face_image, face_info['kps'])
        resized = resize_image(pose_image, self.ctx.pipe_opts.width, self.ctx.pipe_opts.height)

        self.ctx.pipe_opts.image = [resized]
        self.ctx.pipe_opts.width = resized.width
        self.ctx.pipe_opts.height = resized.height
        self.ctx.pipe_opts_extra['controlnet_conditioning_scale'] = 1.0
        self.ctx.pipe_opts_extra['ip_adapter_scale'] = self.ctx.args.ipa_scale[0]

        self.ctx.pipe.load_ip_adapter_instantid(
            hf_download("InstantX/InstantID/ip-adapter.bin", offline=self.ctx.offline))
