import os
import re
import tempfile
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def sorted_nicely(items):
    def convert(text):
        return int(text) if text.isdigit() else text

    def key(item):
        return [convert(part) for part in re.split(r"([0-9]+)", str(item))]

    return sorted(items, key=key)


def parse_case_number(path):
    stem = Path(path).stem
    return int(stem.split("_")[0])


def list_images(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"{folder} is not a valid image directory.")
    images = [
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return [str(path) for path in sorted_nicely(images)]


def load_prompts_csv(prompts_path, required_columns):
    df = pd.read_csv(prompts_path, index_col=False)
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"{prompts_path} is missing columns: {sorted(missing)}")
    return df


def load_clip_model(model_name_or_path, device):
    model = CLIPModel.from_pretrained(model_name_or_path).to(device)
    processor = CLIPProcessor.from_pretrained(model_name_or_path)
    model.eval()
    model.requires_grad_(False)
    return model, processor


def compute_case_clip_scores(image_dir, prompts_path, prompt_column, clip_model_name_or_path, device):
    df = load_prompts_csv(prompts_path, ["case_number", prompt_column])
    device = torch.device(device)
    model, processor = load_clip_model(clip_model_name_or_path, device)

    scores_by_case = defaultdict(list)
    for image_path in list_images(image_dir):
        try:
            case_number = parse_case_number(image_path)
        except ValueError:
            continue

        row = df[df["case_number"] == case_number]
        if row.empty:
            continue

        image = Image.open(image_path).convert("RGB")
        text = str(row.iloc[0][prompt_column])
        inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        scores_by_case[case_number].append(float(outputs.logits_per_image[0][0].detach().cpu()))
        image.close()

    df["clipscore"] = np.nan
    for case_number, scores in scores_by_case.items():
        df.loc[df["case_number"] == case_number, "clipscore"] = float(np.mean(scores))
    return df.dropna(subset=["clipscore"])


def save_clipscore_csv(df, image_dir, prompts_path, save_path):
    if save_path is None:
        return None
    os.makedirs(save_path, exist_ok=True)
    image_name = Path(str(image_dir).rstrip("/")).name
    prompt_name = Path(prompts_path).stem
    out_path = Path(save_path) / f"{image_name}_{prompt_name}_clipscore.csv"
    df.to_csv(out_path, index=False)
    return str(out_path)


def print_grouped_means(df, group_column, metric_column, label):
    print(f"Overall Mean {label}: {df[metric_column].mean():.2f}")
    for group_name, group_df in df.groupby(group_column, sort=False):
        print(f"Mean {label} of {group_name}: {group_df[metric_column].mean():.2f} (cases: {len(group_df)})")


def paired_files_by_group(original_path, edited_path, prompt_path=None, group_column=None, fallback_groups=None):
    original_images = {Path(path).name: path for path in list_images(original_path)}
    edited_images = {Path(path).name: path for path in list_images(edited_path)}
    common_names = sorted_nicely(set(original_images) & set(edited_images))

    case_to_group = {}
    if prompt_path and group_column:
        df = load_prompts_csv(prompt_path, ["case_number", group_column])
        case_to_group = dict(zip(df["case_number"], df[group_column]))

    grouped = defaultdict(list)
    for name in common_names:
        try:
            case_number = parse_case_number(name)
        except ValueError:
            continue

        group = case_to_group.get(case_number)
        if group is None and fallback_groups:
            for start, end, fallback_name in fallback_groups:
                if start <= case_number <= end:
                    group = fallback_name
                    break
        if group is None:
            group = "Unknown"

        grouped[group].append((original_images[name], edited_images[name], name))
    return grouped


def compute_fid_for_pairs(pairs, compute_fid):
    with tempfile.TemporaryDirectory() as original_tmp, tempfile.TemporaryDirectory() as edited_tmp:
        for original, edited, name in pairs:
            shutil.copy(original, os.path.join(original_tmp, name))
            shutil.copy(edited, os.path.join(edited_tmp, name))
        return compute_fid(original_tmp, edited_tmp)
