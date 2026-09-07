import argparse

from cleanfid import fid

from common import compute_fid_for_pairs, paired_files_by_group


CHARACTER_RANGES = [
    (0, 19, "Snoopy"),
    (20, 39, "Mickey"),
    (40, 59, "Spongebob"),
    (60, 79, "Pikachu"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        prog="FIDEveryCharacter",
        description="Compute FID for each character category.",
    )
    parser.add_argument("--original_path", type=str, required=True, help="Path to original images.")
    parser.add_argument("--edited_path", type=str, required=True, help="Path to edited images.")
    parser.add_argument("--prompt_path", type=str, default=None, help="CSV with case_number and category columns.")
    return parser.parse_args()


def main():
    args = parse_args()
    grouped = paired_files_by_group(
        args.original_path,
        args.edited_path,
        prompt_path=args.prompt_path,
        group_column="category" if args.prompt_path else None,
        fallback_groups=CHARACTER_RANGES,
    )

    scores = {}
    for character, pairs in grouped.items():
        if character == "Unknown" or not pairs:
            continue
        score = compute_fid_for_pairs(pairs, fid.compute_fid)
        scores[character] = score
        print(f"FID Score of {character}: {score:.2f} (images: {len(pairs)})")

    if scores:
        print(f"Mean FID over characters: {sum(scores.values()) / len(scores):.2f}")


if __name__ == "__main__":
    main()
