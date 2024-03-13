from huggingface_hub import hf_hub_download
from .base import PluginBase
from safetensors.torch import load_file

def hf_download(fullname, offline=False):
    ns, project, filename = fullname.split('/', 2)
    return hf_hub_download(repo_id=f"{ns}/{project}", filename=filename, local_files_only=offline, resume_download=not offline)

class PluginResAdaptor(PluginBase):

    def setup_args(self, parser):
        parser.add_argument("--res-adaptor", type=float)

    def setup_pipe(self):
        args = self.ctx.args
        pipe = self.ctx.pipe
        if not self.ctx.args.res_adaptor:
            return

        pipe.load_lora_weights(
            hf_download("jiaxiangc/res-adapter/sdxl-i/resolution_lora.safetensors", offline=self.ctx.offline),
            adapter_name="res_adapter"
        )
        self.ctx.loras.append(("res_adapter", args.res_adaptor))
