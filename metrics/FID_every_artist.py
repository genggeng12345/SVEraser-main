import argparse

from cleanfid import fid

from common import compute_fid_for_pairs, paired_files_by_group


ARTIST_RANGES = [
    (0, 19, "Pablo Picasso"),
    (20, 39, "Van Gogh"),
    (40, 59, "Rembrandt"),
    (60, 79, "Andy Warhol"),
    (80, 99, "Caravaggio"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        prog="FIDEveryArtist",
        description="Compute FID for each artist.",
    )
    parser.add_argument("--original_path", type=str, required=True, help="Path to original images.")
    parser.add_argument("--edited_path", type=str, required=True, help="Path to edited images.")
    parser.add_argument("--prompt_path", type=str, default=None, help="CSV with case_number and artist columns.")
    return parser.parse_args()


def main():
    args = parse_args()
    grouped = paired_files_by_group(
        args.original_path,
        args.edited_path,
        prompt_path=args.prompt_path,
        group_column="artist" if args.prompt_path else None,
        fallback_groups=ARTIST_RANGES,
    )

    scores = {}
    for artist, pairs in grouped.items():
        if artist == "Unknown" or not pairs:
            continue
        score = compute_fid_for_pairs(pairs, fid.compute_fid)
        scores[artist] = score
        print(f"FID Score of {artist}: {score:.2f} (images: {len(pairs)})")

    if scores:
        print(f"Mean FID over artists: {sum(scores.values()) / len(scores):.2f}")


if __name__ == "__main__":
    main()
