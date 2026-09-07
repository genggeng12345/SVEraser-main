import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image

from common import DEFAULT_CLIP_MODEL_ID, list_images, load_clip_model, parse_case_number


def parse_args():
    parser = argparse.ArgumentParser(
        prog="CLIPAccuracyEveryCharacter",
        description="Compare erased and target CLIP scores and report accuracy for each character category.",
    )
    parser.add_argument("--img_path", type=str, required=True, help="Path to generated images.")
    parser.add_argument("--erased_concept", type=str, default=None, help="Erased concept prompt.")
    parser.add_argument("--target_concept", type=str, default=None, help="Target concept prompt.")
    parser.add_argument("--prompt_path", type=str, default=None, help="CSV with per-case prompts and category.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--clip_model_name_or_path", type=str, default=DEFAULT_CLIP_MODEL_ID)
    return parser.parse_args()


def score_pair(model, processor, image_path, erased_prompt, target_prompt, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(
        text=[str(erased_prompt), str(target_prompt)],
        images=image,
        return_tensors="pt",
        padding=True,
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = outputs.logits_per_image[0].detach().cpu().numpy()
    image.close()
    return float(scores[0]), float(scores[1])


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, processor = load_clip_model(args.clip_model_name_or_path, device)
    image_paths = list_images(args.img_path)

    grouped_correct = defaultdict(list)

    if args.prompt_path:
        df = pd.read_csv(args.prompt_path)
        required = {"case_number", "category", "erased_concept", "target_concept"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{args.prompt_path} is missing columns: {sorted(missing)}")
        case_to_row = {int(row.case_number): row for row in df.itertuples(index=False)}

        for image_path in image_paths:
            try:
                case_number = parse_case_number(image_path)
            except ValueError:
                continue
            row = case_to_row.get(case_number)
            if row is None:
                continue
            erased_score, target_score = score_pair(
                model, processor, image_path, row.erased_concept, row.target_concept, device
            )
            grouped_correct[str(row.category)].append(erased_score > target_score)
    else:
        if args.erased_concept is None or args.target_concept is None:
            raise ValueError("Provide --prompt_path or both --erased_concept and --target_concept.")
        for image_path in image_paths:
            erased_score, target_score = score_pair(
                model, processor, image_path, args.erased_concept, args.target_concept, device
            )
            grouped_correct["all"].append(erased_score > target_score)

    all_values = [value for values in grouped_correct.values() for value in values]
    print(f"Overall CLIP accuracy: {np.mean(all_values):.2f}")
    for category, values in grouped_correct.items():
        print(f"{category} CLIP accuracy: {np.mean(values):.2f} (images: {len(values)})")


if __name__ == "__main__":
    main()
