# Model Weights

This repository does not include large model weights.

## Stable Diffusion

The training and generation scripts use official Hugging Face model ids by default:

- SD 1.4: `CompVis/stable-diffusion-v1-4`
- SD 2.1 base: `stabilityai/stable-diffusion-2-1-base`

You can also pass a local diffusers-format model path with `--pretrained_model_name_or_path` or `--sd_path`.

## CLIP Scoring

`ClipScore_*.py`, `CLIPAccuracy_every_character.py`, and `generate-images_sd2.py` use official CLIP by default:

- `openai/clip-vit-large-patch14`

These scripts load CLIP through Hugging Face `transformers`:

```python
CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
```

No local CLIP model path is hard-coded.

## CSD

`metrics/CSD_every_artist.py` requires two external weight files that are not included:

- CSD checkpoint: `checkpoint.pth`
- OpenAI CLIP ViT-L/14 backbone: `ViT-L-14.pt`

Run it by explicitly passing both paths:

```bash
python metrics/CSD_every_artist.py \
  --ori_img_path /path/to/original_images \
  --tar_img_path /path/to/edited_images \
  --prompt_path /path/to/prompts.csv \
  --csd_checkpoint /path/to/checkpoint.pth \
  --clip_backbone_path /path/to/ViT-L-14.pt
```

These files are large and should be provided via a download link or Git LFS release asset.

## NudeNet And FID

`nudenet_count.py` may download or cache NudeNet/ONNXRuntime model assets on first use.

`FID_*.py` uses `clean-fid`, which may download or cache Inception/FID assets on first use.
