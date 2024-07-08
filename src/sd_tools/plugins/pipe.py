import psutil
from dataclasses import asdict
from .base import PluginBase

class PluginPipe(PluginBase):

    def setup_pipe(self):

        if self.ctx.debug:
            print(f"{self.ctx.pipeline} {self.ctx.model}")

        opts = {**asdict(self.ctx.pipeline_opts), **self.ctx.pipeline_opts_extra}

        if self.ctx.model.endswith(".safetensors"):
            self.ctx.pipe = self.ctx.pipeline.from_single_file(
                self.ctx.model,
                **opts
            ).to(self.ctx.device)
        else:
            self.ctx.pipe = self.ctx.pipeline.from_pretrained(
                self.ctx.model,
                **opts
            ).to(self.ctx.device)

        # enable attention slicing if RAM < 64 GB
        # if (available_ram := psutil.virtual_memory().available / (1024**3)) < 64:
        #     print(f"available ram {available_ram} < 64")
        #     self.ctx.pipe.enable_attention_slicing()
