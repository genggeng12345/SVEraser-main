import argparse
from cleanfid import fid



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='FID',
        description='Compute FID score between two directories')
    parser.add_argument('--original_path', help='path to original image', type=str, required=True)
    parser.add_argument('--edited_path', help='path to edited image', type=str, required=True)

    args = parser.parse_args()
    score = fid.compute_fid(args.original_path, args.edited_path)
    print(f'Average FID score: {score}')
