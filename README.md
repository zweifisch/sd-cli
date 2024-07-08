# sd-tools

image generation from the command line and web UI

![web](https://github.com/zweifisch/sd-tools/assets/447862/3855dbd1-65ba-4721-ad44-0af6d79eb0c0)

```shell
pip install sd-tools
```

launch the Web UI

```shell
sdxl \
 --model Lykon/dreamshaper-xl-lightning \
 --scheduler 'DPM++ SDE Karras' \
 --listen 8484
```

visit [http://127.0.0.1:8484](http://127.0.0.1:8484)

more detailed instructions:

- [commandline usage](doc/cli.md)
- [http api](doc/api.md)
