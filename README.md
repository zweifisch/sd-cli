# sd-cli

```shell
pip install sd-cli
```

## usage

```shell
sd --model 'SG161222/RealVisXL_V4.0_Lightning'\
 --prompt 'portrait of a man, materpiece' \
 --steps 8 \
 --size 1024x576 \
 --scheduler 'DPM++ SDE Karras' \
 --cfg 3
```

### speed up generation

with `--lcm` `--tcd` or `--lightning`

```shell
sd \
 --prompt 'portrait of a man, materpiece' \
 --steps 4 \
 --size 1024x576 \
 --scheduler 'DPM++ SDE Karras' \
 --lightning 1
```

### IP-Adapter Plus

### IP-Adapter Plus Face

### Photo Maker

```shell
pip install sd-cli[photomaker]
```

```shell
sd \
 --prompt 'portrait of a man img, materpiece' \
 --steps 8 \
 --size 1024x576 \
 --scheduler 'DPM++ SDE Karras' \
 --photo-maker ./photos
```

reference images can also be specified by filenames

`--photo-maker 1.png 2.png`

### FaceID

```shell

```

### Canny

```shell

```

### Pose

```shell

```
