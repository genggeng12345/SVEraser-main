import argparse

from common import DEFAULT_CLIP_MODEL_ID, compute_case_clip_scores, save_clipscore_csv


def parse_args():
    parser = argparse.ArgumentParser(
        prog="CLIPScore",
        description="Compute average CLIPScore for all COCO-style prompts.",
    )
    parser.add_argument("--img_path", type=str, required=True, help="Path to generated images.")
    parser.add_argument("--prompts_path", type=str, required=True, help="CSV with case_number and prompt columns.")
    parser.add_argument("--save_path", type=str, default=None, help="Directory to save per-case CSV.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--clip_model_name_or_path", type=str, default=DEFAULT_CLIP_MODEL_ID)
    return parser.parse_args()


def main():
    args = parse_args()
    df = compute_case_clip_scores(
        image_dir=args.img_path,
        prompts_path=args.prompts_path,
        prompt_column="prompt",
        clip_model_name_or_path=args.clip_model_name_or_path,
        device=args.device,
    )
    save_clipscore_csv(df, args.img_path, args.prompts_path, args.save_path)
    print(f"Mean clipscore: {df['clipscore'].mean():.2f}")


if __name__ == "__main__":
    main()
