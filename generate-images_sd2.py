import argparse
import gc
import os
import time
from pathlib import Path

import pandas as pd
import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor
from safetensors.torch import load_file
from svdiff_pytorch import load_unet_for_svdiff
from transformers import CLIPModel, CLIPTokenizer


DEFAULT_SD2_MODEL_ID = "stabilityai/stable-diffusion-2-1-base"
DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-large-patch14"


def num_open_fds():
    try:
        return len(os.listdir("/proc/self/fd"))
    except Exception:
        return -1


def text_tokenize(tokenizer: CLIPTokenizer, prompts):
    return tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )


def text_encode(clip_model: CLIPModel, text_inputs):
    text_inputs = {
        key: value.to(clip_model.device)
        for key, value in text_inputs.items()
    }
    return clip_model.get_text_features(**text_inputs)


def encode_prompts(
    tokenizer: CLIPTokenizer,
    clip_model: CLIPModel,
    prompts,
    return_tokens: bool = False,
):
    text_inputs = text_tokenize(tokenizer, prompts)
    text_embeddings = text_encode(clip_model, text_inputs)

    if return_tokens:
        return text_embeddings, torch.unique(text_inputs.input_ids, dim=1)
    return text_embeddings


def load_state_dict(file_name, dtype):
    if os.path.splitext(str(file_name))[1] == ".safetensors":
        sd = load_file(str(file_name))
    else:
        sd = torch.load(str(file_name), map_location="cpu")

    for key in list(sd.keys()):
        if isinstance(sd[key], torch.Tensor):
            sd[key] = sd[key].to(dtype=dtype, device="cpu")
    return sd


def parse_precision(precision: str) -> torch.dtype:
    precision = precision.lower()
    if precision in ["fp32", "float32"]:
        return torch.float32
    if precision in ["fp16", "float16"]:
        return torch.float16
    if precision in ["bf16", "bfloat16"]:
        return torch.bfloat16
    raise ValueError(f"Invalid precision type: {precision}")


def calculate_matching_score(prompt_embeds, erased_prompt_embeds):
    split_prompt_embeds = torch.split(prompt_embeds, 1, dim=0)
    scores = None

    for prompt_embed in split_prompt_embeds:
        clipcos = torch.cosine_similarity(
            prompt_embed.flatten(1),
            erased_prompt_embeds.flatten(1),
            dim=-1,
        ).detach().cpu()
        if scores is None:
            scores = clipcos
        else:
            scores = torch.where(clipcos > scores, clipcos, scores)
    return scores


def generate_images(
    sd_path,
    clip_model_name_or_path,
    prompts_path,
    save_name,
    spectral_shifts_paths,
    erased_prompts,
    csv_path,
    device="cuda:0",
    guidance_scale=7.5,
    image_size=768,
    ddim_steps=50,
    num_samples=1,
    micro_batch_size=1,
    from_case=0,
    till_case=1000000,
    precision="fp32",
    spectral_shifts_scale=1.0,
    selection_threshold=0.8,
    use_slicing=False,
):
    device = torch.device(device)
    weight_dtype = parse_precision(precision)

    unet = load_unet_for_svdiff(sd_path, subfolder="unet", torch_dtype=weight_dtype)
    pipe = StableDiffusionPipeline.from_pretrained(
        sd_path,
        unet=unet,
        torch_dtype=weight_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=False)

    if use_slicing:
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()

    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name_or_path)
    clip_model = CLIPModel.from_pretrained(
        clip_model_name_or_path,
        torch_dtype=weight_dtype,
    ).to(device)
    clip_model.eval()
    clip_model.requires_grad_(False)

    for module in unet.modules():
        if hasattr(module, "set_scale"):
            module.set_scale(scale=spectral_shifts_scale)

    original_state_dict = {
        key: value.detach().cpu().clone()
        for key, value in unet.state_dict().items()
    }

    df = pd.read_csv(prompts_path)
    required_columns = {"case_number", "prompt", "evaluation_seed"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"prompts CSV is missing columns: {sorted(missing_columns)}")

    os.makedirs(save_name, exist_ok=True)
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    with torch.no_grad():
        erased_prompt_embeds = encode_prompts(
            clip_tokenizer,
            clip_model,
            erased_prompts,
            return_tokens=False,
        )

    svds = [load_state_dict(path, dtype=weight_dtype) for path in spectral_shifts_paths]
    if len(svds) != len(erased_prompts):
        raise ValueError(
            "The number of --spectral_shifts_paths must match the number of --erased_prompts."
        )

    head = ["case_number", "prompt"] + [f"score_{prompt}" for prompt in erased_prompts] + [
        "selected_erased_prompts",
        "selected_scores",
    ]
    pd.DataFrame(columns=head).to_csv(csv_path, index=False)

    vae_scale_factor = getattr(pipe, "vae_scale_factor", None)
    if vae_scale_factor is None:
        vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    if image_size % vae_scale_factor != 0:
        raise ValueError(
            f"image_size={image_size} is not divisible by vae_scale_factor={vae_scale_factor}"
        )

    latent_h = image_size // vae_scale_factor
    latent_w = image_size // vae_scale_factor

    print(f"Using CLIP scoring model: {clip_model_name_or_path}")
    print(f"Using SD2 generation model: {sd_path}")
    print(f"Using vae_scale_factor={vae_scale_factor}, latent size=({latent_h}, {latent_w})")
    print(f"precision={precision}, num_samples={num_samples}, micro_batch_size={micro_batch_size}")

    idx = 0
    all_select_time = 0.0
    all_generate_time = 0.0
    rows_data = []

    for row_idx, row in enumerate(df.itertuples(index=False), start=0):
        start_time = time.time()
        case_number = int(row.case_number)
        if not (from_case <= case_number <= till_case):
            continue

        prompt = str(row.prompt)
        with torch.no_grad():
            prompt_embeds = encode_prompts(
                clip_tokenizer,
                clip_model,
                [prompt],
                return_tokens=False,
            )
            scores = calculate_matching_score(prompt_embeds, erased_prompt_embeds)
        scores_list = scores.flatten().tolist()

        selected = [
            (erased_prompt, score, svd)
            for erased_prompt, score, svd in zip(erased_prompts, scores_list, svds)
            if score > selection_threshold
        ]

        print(f"case_number={case_number}, multipliers={scores_list}")

        row_dict = {
            "case_number": case_number,
            "prompt": prompt,
        }
        for erased_prompt, score in zip(erased_prompts, scores_list):
            row_dict[f"score_{erased_prompt}"] = score
        row_dict["selected_erased_prompts"] = str([item[0] for item in selected])
        row_dict["selected_scores"] = str([item[1] for item in selected])
        rows_data.append(row_dict)

        new_state_dict = {key: value.clone() for key, value in original_state_dict.items()}
        for _, multiplier, svd in selected:
            for key, value in svd.items():
                if key in new_state_dict:
                    new_state_dict[key] += value.to(
                        dtype=new_state_dict[key].dtype,
                        device="cpu",
                    ) * float(multiplier)

        unet.load_state_dict(new_state_dict, strict=False)

        end_select_time = time.time()
        torch.cuda.empty_cache()

        base_seed = int(row.evaluation_seed)
        generator = torch.Generator(device=device).manual_seed(base_seed)
        all_latents = randn_tensor(
            (
                num_samples,
                pipe.unet.config.in_channels,
                latent_h,
                latent_w,
            ),
            generator=generator,
            device=device,
            dtype=weight_dtype,
        )

        images = []
        with torch.inference_mode():
            for start in range(0, num_samples, micro_batch_size):
                end = min(start + micro_batch_size, num_samples)
                batch_latents = all_latents[start:end]
                batch_prompts = [prompt] * (end - start)

                batch_images = pipe(
                    batch_prompts,
                    height=image_size,
                    width=image_size,
                    guidance_scale=guidance_scale,
                    num_inference_steps=ddim_steps,
                    latents=batch_latents,
                    generator=generator,
                    negative_prompt=None,
                ).images
                images.extend(batch_images)

                del batch_images, batch_latents
                torch.cuda.empty_cache()

        for image_index, image in enumerate(images):
            out_path = os.path.join(save_name, f"{case_number}_{image_index}.png")
            image.save(out_path)
            image.close()

        del images, all_latents
        torch.cuda.empty_cache()

        if (row_idx + 1) % 10 == 0:
            pd.DataFrame(rows_data[-10:]).to_csv(csv_path, mode="a", header=False, index=False)

        if row_idx % 100 == 0:
            gc.collect()
            print("open_fds:", num_open_fds())

        end_generate_time = time.time()
        all_select_time += end_select_time - start_time
        all_generate_time += end_generate_time - end_select_time
        idx += 1

    remain = len(rows_data) % 10
    if remain:
        pd.DataFrame(rows_data[-remain:]).to_csv(csv_path, mode="a", header=False, index=False)

    all_select_time = all_select_time / max(idx, 1)
    all_generate_time = all_generate_time / max(idx, 1)
    print(f"Average time for selecting spectral shifts: {all_select_time:.4f} s")
    print(f"Average time for generating images: {all_generate_time:.4f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generateImagesSD2",
        description="Generate images with SD2 and SVDiff spectral shifts, scored by a separate CLIP text model.",
    )
    parser.add_argument(
        "--sd_path",
        type=str,
        default=DEFAULT_SD2_MODEL_ID,
        help="Path to a diffusers-format SD2 model or a Hugging Face model id.",
    )
    parser.add_argument(
        "--clip_model_name_or_path",
        type=str,
        default=DEFAULT_CLIP_MODEL_ID,
        help="Official CLIP model id or local CLIP path used only for score calculation.",
    )
    parser.add_argument("--prompts_path", type=str, required=True, help="Path to CSV file with prompts.")
    parser.add_argument("--save_name", type=str, required=True, help="Folder where images are saved.")
    parser.add_argument(
        "--spectral_shifts_paths",
        required=True,
        nargs="+",
        help="Paths to spectral_shifts.safetensors files.",
    )
    parser.add_argument(
        "--erased_prompts",
        required=True,
        nargs="+",
        help="Erased prompts corresponding to --spectral_shifts_paths.",
    )
    parser.add_argument("--csv_path", type=str, required=True, help="CSV path for CLIP matching scores.")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device to run on.")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--spectral_shifts_scale", type=float, default=1.0)
    parser.add_argument("--selection_threshold", type=float, default=0.8)
    parser.add_argument("--image_size", type=int, default=768)
    parser.add_argument("--till_case", type=int, default=1000000)
    parser.add_argument("--from_case", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--precision", type=str, default="fp32", help="fp32 / fp16 / bf16")
    parser.add_argument("--use_slicing", action="store_true")
    args = parser.parse_args()

    generate_images(
        sd_path=args.sd_path,
        clip_model_name_or_path=args.clip_model_name_or_path,
        prompts_path=args.prompts_path,
        save_name=args.save_name,
        spectral_shifts_paths=[Path(path) for path in args.spectral_shifts_paths],
        erased_prompts=[str(prompt) for prompt in args.erased_prompts],
        csv_path=args.csv_path,
        precision=args.precision,
        device=args.device,
        guidance_scale=args.guidance_scale,
        image_size=args.image_size,
        ddim_steps=args.ddim_steps,
        num_samples=args.num_samples,
        micro_batch_size=args.micro_batch_size,
        from_case=args.from_case,
        till_case=args.till_case,
        spectral_shifts_scale=args.spectral_shifts_scale,
        selection_threshold=args.selection_threshold,
        use_slicing=args.use_slicing,
    )
