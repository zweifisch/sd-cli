# sd-tools

```shell
pip install sd-tools
```

## Basics

```shell
sdxl 'locomotive comming' --size 512
```

If the model can't be downloaded, try using a huggingface mirror:

```shell
export HF_MIRROR=https://hf-mirror.com
```

To get better result, use more steps and higher resolution:

```shell
sdxl 'locomotive comming' --size 720 --steps 4
```

Generate more with different resolution:

```shell
sdxl 'locomotive comming' --size 1024x576 --steps 4 --count 4
```

### Interactive Mode

The loading of the model can task some time, use `-i` to enter interactive mode, keeping the model loaded:

```shell
sdxl 'locomotive comming' --size 1024x576 --steps 4 -i
```

Edited prompt and press enter:

```
> locomotive comming
```

`size`, `count` and `cfg` can also be set on the fly:

```
> locomotive comming size=1024
```

### Custom Models

To use a specific model:

```shell
sdxl --model Lykon/dreamshaper-xl-lightning \
 --steps 6 \
 --size 1024x576 \
 --scheduler 'DPM++ SDE Karras' \
 --cfg 2 \
 'locomotive comming'
```

More models can be found on [huggingface](https://huggingface.co/models?pipeline_tag=text-to-image&sort=trending) and [civitai](https://civitai.com/models), models with Lightning or Turbo can generate image in less than 8 steps.

For models without fp16 variant and safetensor format, add `--no-fp16` and `--no-safetensor`

```shell
sdxl --model RunDiffusion/Juggernaut-XL-Lightning \
 --cfg 1.5 \
 --steps 6 \
 --size 832x1216 \
 --scheduler 'DPM++ SDE' \
 --no-fp16 --no-safetensor \
 'locomotive comming'
```

To use models downloaded from civitai:

```shell
sdxl --model ./model.safetensors 'locomotive comming'
```

### Loras

Loras are like plugins for the base model, multiple loras can be used:

```shell
sdxl \
 --model SG161222/RealVisXL_V4.0_Lightning \
 --steps 8 \
 --size 1024x576 \
 --loras ./lora1.safetensors ./lora2.safetensors:0.8 \
 'locomotive comming'
```

### Speed Up Generation

Using `--lcm`, `--tcd` or `--lightning` to speed up generation, not necessary for turbo/lightning models:

```shell
sdxl \
 --model SG161222/RealVisXL_V4.0 \
 --steps 4 \
 --size 1024x576 \
 --tcd 1 \
 'locomotive comming'
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

## Style and Face

### Photo Maker

```shell
pip install sd-tools[photomaker]
```

use `img` to indicate the reference target

```shell
sdxl \
 --model Lykon/dreamshaper-xl-lightning \
 --steps 8 \
 --size 1024 \
 --scheduler 'DPM++ SDE Karras' \
 --photo-maker ./photos \
 'portrait of a man img'
```

reference images can also be specified by filenames

`--photo-maker 1.png 2.png`

### IP-Adapter Plus

style reference

```shell
sdxl 'Mario' --ipa-plus ./styles/
```

face reference

```shell
sdxl 'Mario' --ipa-plus-face face1.png face2.png
```

combined

```shell
sdxl 'Mario' --ipa-plus ./styles --ipa-plus-face ./faces
```

### IP-Adapter FaceID Plus

```shell
pip install sd-tools[faceid]
```

```shell
sdxl 'Mario' --ipa-faceid face.png
```

### InstantID

```shell
pip install sd-tools[faceid]
```

```shell
sdxl 'Mario' --instanceid face.png
```

## HTTP Server

```shell
sdxl 'Mario' --listen '127.0.0.1:8800'
```

invoke using curl:

```shell
curl '127.0.0.1:8800?prompt=shell&seed=1' > image.webp
```

## More Examples

```shell
sdxl --model RunDiffusion/Juggernaut-XL-Lightning \
 --prompt 'portrait of a woman' \
 --cfg 1.5 \
 --steps 5 \
 --size 832x1216 \
 --scheduler 'DPM++ SDE' \
 --no-fp16 --no-safetensor
```

## SD 1.5

TBD
