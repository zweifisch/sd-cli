import os
from typing import List
from PIL import Image
from huggingface_hub import hf_hub_download
from diffusers.utils import load_image

def resize_image(image, width, height) -> Image.Image:
    width0, height0 = image.size

    fit_width = width0 / height0 > width / height

    new_width = width if fit_width else int(width0 * height / height0)
    new_height = height if not fit_width else int(height0 * width / width0)

    resized = image.resize((new_width // 8 * 8, new_height // 8 * 8))
    return resized

    new_image = Image.new("RGB", (width, height), (0, 0, 0))
    new_image.paste(resized, ((width - new_width) // 2, (height - new_height) // 2))

    return new_image

def hf_download(fullname, offline=False):
    ns, project, filename = fullname.split('/', 2)
    return hf_hub_download(repo_id=f"{ns}/{project}", filename=filename, local_files_only=offline, resume_download=not offline)

def load_images(locations: List[str]) -> List[Image.Image]:
    if locations is None:
        return []
    if '.' in os.path.basename(locations[0]):
        return [load_image(x) for x in locations]
    return [load_image(x) for x in sorted([os.path.join(locations[0], basename) for basename in os.listdir(locations[0])])]

def remove_none(input_dict):
    return {k: v for k, v in input_dict.items() if v is not None}
