import os
import sys
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# 将父目录添加到 sys.path
sys.path.append(parent_dir)
import argparse
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from model import CSD_CLIP, convert_state_dict
from torchvision import transforms
import torchvision.transforms.functional as F
import pandas as pd
from collections import defaultdict


class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = pre_data_process()

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image': image}

    def __len__(self):
        return len(self.data)

def prepare_model(device, csd_checkpoint, clip_backbone_path):
    # init model
    if clip_backbone_path is None or not os.path.exists(clip_backbone_path):
        raise FileNotFoundError(
            "CSD requires a CLIP ViT-L/14 backbone checkpoint. "
            "Pass --clip_backbone_path /path/to/ViT-L-14.pt. See MODEL_WEIGHTS.md."
        )
    model = CSD_CLIP("vit_large", "default", model_path=clip_backbone_path)
    # load model
    if csd_checkpoint is None or not os.path.exists(csd_checkpoint):
        raise FileNotFoundError(
            "CSD requires checkpoint.pth. "
            "Pass --csd_checkpoint /path/to/checkpoint.pth. See MODEL_WEIGHTS.md."
        )
    checkpoint = torch.load(csd_checkpoint, map_location="cpu")
    state_dict = convert_state_dict(checkpoint['model_state_dict'])
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    return model

def pre_data_process():
    # normalization
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    preprocess = transforms.Compose([
                    transforms.Resize(size=224, interpolation=F.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])
    return preprocess

def get_all_images(folder_path, image_extensions=None):
    # 默认支持的图片格式
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']

    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        raise ValueError(f"指定路径 {folder_path} 不是有效文件夹。")
    # 搜索图片
    image_paths = [str(file) for file in folder_path.rglob("*") if file.suffix.lower() in image_extensions]

    print(f"在 {folder_path} 中找到 {len(image_paths)} 张图片。")

    image_paths.sort(key=extract_numbers)

    return image_paths

# 提取文件名中的 a_b 数字对
def extract_numbers(path):
    match = re.search(r'(\d+)_(\d+)', Path(path).stem)
    if not match:
        raise ValueError(f"文件名 {path} 不包含 a_b 数字对。")
    return int(match.group(1)), int(match.group(2))

def check_image_filenames_identical(ori_imgs, tar_imgs):
    # 提取文件名，不考虑路径
    ori_filenames = {Path(img).name for img in ori_imgs}  # 使用 set 去除重复
    tar_filenames = {Path(img).name for img in tar_imgs}  # 使用 set 去除重复
    # 比较文件名
    if ori_filenames == tar_filenames:
        print("The two folder images correspond to each other, cal available !")
    else:
        raise ValueError("ori_imgs 和 tar_imgs 中的文件名不一致。")


def extract_all_images(images, model, device, batch_size=64, num_workers=8):
    data = torch.utils.data.DataLoader(
        CLIPImageDataset(images),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_image_features = []
    with torch.no_grad():
        for b in tqdm(data):
            b = b['image'].to(device)
            #print(b.shape)
            _, content_output, style_output = model(b)
            all_image_features.append(style_output.cpu().numpy())
    all_image_features = np.vstack(all_image_features)
    # print(all_image_features.shape)
    return all_image_features


def cal_CSD_Score(ori_img_path, tar_img_path, device, csd_checkpoint, clip_backbone_path):
    model = prepare_model(device, csd_checkpoint, clip_backbone_path)
    ori_images = get_all_images(ori_img_path)
    tar_images = get_all_images(tar_img_path)
    check_image_filenames_identical(ori_images, tar_images)
    ori_features = extract_all_images(ori_images, model, device)
    tar_features = extract_all_images(tar_images, model, device)
    full_sim = ori_features@tar_features.T
    sim = full_sim.diagonal()
    return sim.mean()

def parse_args():
    parser = argparse.ArgumentParser("metric", add_help=False)
    parser.add_argument("--ori_img_path", type=str, required=True,
                        help="the folder to ori images")
    parser.add_argument("--tar_img_path", type=str, required=True,
                        help="the folder to tar images")
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--prompt_path", type=str, required=False,default=None,
                        help="the path to prompt csv file")
    parser.add_argument("--csd_checkpoint", type=str, default=None,
                        help="path to the CSD checkpoint.pth file")
    parser.add_argument("--clip_backbone_path", type=str, default=None,
                        help="path to the OpenAI CLIP ViT-L-14.pt checkpoint")

    return parser.parse_args()


def get_same_style(prompt_path, image_paths):
    df = pd.read_csv(prompt_path)
    artist_dict = defaultdict(list)
    # 创建 case_number -> artist 映射
    case_to_artist = dict(zip(df["case_number"], df["artist"]))
    # 按 artist 分组
    for img_path in image_paths:
        try:
            case_number, _ = extract_numbers(img_path)
            artist = case_to_artist.get(case_number, "Unknown")  # 若找不到，归为 "Unknown"
            artist_dict[artist].append(img_path)
        except ValueError:
            print(f"无法从文件名中提取数字对：{img_path}")

    return artist_dict


if __name__ == '__main__':
    args = parse_args()
    # 没有指定 prompt_path 则计算全部图片的 CSD Score
    if not args.prompt_path:
        model = prepare_model(args.device, args.csd_checkpoint, args.clip_backbone_path)
        ori_images = get_all_images(args.ori_img_path)
        tar_images = get_all_images(args.tar_img_path)
        check_image_filenames_identical(ori_images, tar_images)
        ori_features = extract_all_images(ori_images, model, args.device)
        tar_features = extract_all_images(tar_images, model, args.device)
        full_sim = ori_features @ tar_features.T
        sim = full_sim.diagonal().mean()
        print(f"CSD Score: {sim:.2f}")  # 修改这里，保留两位小数
    else:
        model = prepare_model(args.device, args.csd_checkpoint, args.clip_backbone_path)
        ori_images = get_all_images(args.ori_img_path)
        tar_images = get_all_images(args.tar_img_path)
        ori_images_dict = get_same_style(args.prompt_path, ori_images)
        tar_images_dict = get_same_style(args.prompt_path, tar_images)

        res = {}
        for artist, paths in ori_images_dict.items():
            print(f"Artist: {artist}")
            print(f"Images: {len(paths)}")
            if artist == "Unknown":
                print(f"Skipping artist {artist}")
                continue
            ori_images = paths
            tar_images = tar_images_dict.get(artist, [])
            check_image_filenames_identical(ori_images, tar_images)
            ori_features = extract_all_images(ori_images, model, args.device)
            tar_features = extract_all_images(tar_images, model, args.device)
            full_sim = ori_features @ tar_features.T
            sim = full_sim.diagonal()
            res[artist] = sim.mean()

        for artist, sim in res.items():
            print(f"{artist} CSD Score: {sim:.2f}")  # 修改这里，保留两位小数
