# sd-tools

```shell
pip install sd-tools
```

## Basics

```shell
sdxl 'locomotive'
```

If the model can't be downloaded, try using a huggingface mirror:

```shell
export HF_MIRROR=https://hf-mirror.com
```

To get better result, use more steps and higher resolutions:

```shell
sdxl 'locomotive' --size 720 --steps 4
```

Generate more with `--count`:

```shell
sdxl 'locomotive' --size 1024x576 --steps 4 --count 3
```

By default, images are save to `output/{seed}-{time}.webp`, which can be customized via `-o`:

```shell
sdxl 'locomotive' -o 'output/{seed}-{size}-{cfg}-{time}.png'
```

### Interactive Mode

The loading of the model can task some time, use `-i` to enter interactive mode, keeping the model loaded:

```shell
sdxl 'locomotive' --size 1024x576 --steps 4 -i
```

Edited prompt and press enter:

```
> locomotive
```

`size`, `count`, `steps` and `cfg` can also be set on the fly:

```
> locomotive size=1024 cfg=1.9
```

### Custom Models

To use a specific model:

```shell
sdxl 'locomotive' \
 --model Lykon/dreamshaper-xl-lightning \
 --steps 6 \
 --size 1024x576 \
 --scheduler 'DPM++ SDE Karras' \
 --cfg 2 \
```

More models can be found on [huggingface](https://huggingface.co/models?pipeline_tag=text-to-image&sort=trending) and [civitai](https://civitai.com/models), models with Lightning or Turbo can generate image in less than 8 steps.

For models without fp16 variant or safetensor format, add `--no-fp16` or `--no-safetensor`

```shell
sdxl 'locomotive' \
 --model RunDiffusion/Juggernaut-XL-Lightning \
 --cfg 1.5 \
 --steps 6 \
 --size 832x1216 \
 --scheduler 'DPM++ SDE' \
 --no-fp16 --no-safetensor
```

To use models downloaded from civitai:

```shell
sdxl 'locomotive' --model ./model.safetensors
```

### Loras

Loras are like plugins for the base model, multiple loras can be used:

```shell
sdxl 'locomotive' \
 --model SG161222/RealVisXL_V4.0_Lightning \
 --steps 8 \
 --size 1024x576 \
 --loras ./lora1.safetensors ./lora2.safetensors:0.8 \
```

### Speed Up Generation

Using `--lcm`, `--tcd` or `--lightning` to speed up generation, not necessary for turbo/lightning models:

```shell
sdxl 'locomotive' \
 --model SG161222/RealVisXL_V4.0 \
 --steps 4 \
 --size 1024x576 \
 --tcd 1 \
```

## More Controlling

### Canny

```shell
sdxl 'protrait of a man' --canny photo.jpg --canny-low 100 --canny-high 200
```

### Depth

```shell
sdxl 'protrait of a man' --depth photo.jpg
```

### Pose

```shell
sdxl 'protrait of a man' --pose photo.jpg
```

### Inpainting

```shell
sdxl 'helmet' \
 --inpaint test/tesla.webp \
 --inpaint-mask mask.png \
 --steps 40 \
 -i \
 -o output/preview.png
```

Open mask.png, paint the area to be modified, close image editor, then press Enter to start inpainting.

## Style and Face

### Photo Maker

```shell
pip install 'sd-tools[photomaker]'
```

use `img` to indicate the reference target

```shell
sdxl 'man img holding a toy car' \
 --model Lykon/dreamshaper-xl-lightning \
 --steps 6 \
 --cfg 2 \
 --photo-maker test/tesla.webp \
 --scheduler 'DPM++ SDE Karras'
```

multiple reference images can be provided:

`--photo-maker 1.png 2.png`

### IP-Adapter Plus

style reference, multiple images can be used:

```shell
sdxl 'portrait of a man' \
 --model Lykon/dreamshaper-xl-lightning \
 --scheduler 'DPM++ SDE Karras' \
 --steps 8 --size 1024x576 \
 --ipa-plus test/ghibli \
 --ipa-plus-scale 0.7 \
 --seed 0
```

face reference, multiple images can be used:

```shell
sdxl 'portrait of a man' \
 --model Lykon/dreamshaper-xl-lightning \
 --scheduler 'DPM++ SDE Karras' \
 --steps 8 --size 1024x576 \
 --ipa-plus-face test/tesla.webp \
 --ipa-plus-scale 0.4 \
 --seed 0
```

combined

```shell
sdxl 'portrait of a man' \
 --model Lykon/dreamshaper-xl-lightning \
 --scheduler 'DPM++ SDE Karras' \
 --steps 8 --size 1024x576 \
 --ipa-plus test/ghibli \
 --ipa-plus-face test/tesla.webp \
 --ipa-plus-scale 0.7 0.4 \
 --cfg 2 \
 --seed 0
```

### IP-Adapter FaceID Plus

```shell
pip install 'sd-tools[faceid]'
```

```shell
sdxl 'man walking on the moon' \
 --model Lykon/dreamshaper-xl-lightning \
 --scheduler 'DPM++ SDE Karras' \
 --steps 6 \
 --cfg 2 \
 --ipa-faceid-plus test/tesla.webp
```

### InstantID

```shell
pip install 'sd-tools[faceid]'
```

```shell
sdxl 'man walking on the moon' \
 --model Lykon/dreamshaper-xl-lightning \
 --scheduler 'DPM++ SDE Karras' \
 --steps 6 \
 --cfg 2 \
 --instantid test/tesla.webp
```

## HTTP Server

```shell
sdxl 'man walking on the moon' --listen 127.0.0.1:8800
```

Invoking using curl:

```shell
curl '127.0.0.1:8800?prompt=shell&seed=1' > image.webp
```

## Live Preview

```shell
sdxl 'locomotive' --size 512 -i -o output/preview.png
```

Open output folder in Finder, switch to gallery mode, once generated, image preview will be updated.

You can even set a count to generate more images:

```shell
> locomotive count=3
```

To keep the generated images, pass an additional ouput path:

```shell
sdxl 'locomotive' --size 512 -i -o output/preview.png 'output/{seed}.webp'
```

## More Examples

### Juggernaut

```shell
sdxl --prompt 'portrait of a woman' \
 --model RunDiffusion/Juggernaut-XL-Lightning \
 --cfg 1.9 \
 --steps 6 \
 --size 832x1216 \
 --scheduler 'DPM++ SDE' \
 --no-fp16 --no-safetensor
```

## SD 1.5

```shell
sd -h
```
