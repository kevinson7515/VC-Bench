import os
import json
import logging
import subprocess
import numpy as np
import clip
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from tqdm import tqdm
current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None
from benchmark.utils import load_video, clip_transform
from benchmark.distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)

import warnings
warnings.filterwarnings("ignore")


def get_video_files(folder_path):
    """
    获取文件夹中所有视频文件的路径列表
    
    参数:
        folder_path (str): 视频文件夹路径
        
    返回:
        list: 视频文件路径列表（绝对路径）
    """
    # 支持的视频扩展名（可根据需要扩展）
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    
    video_files = []
    
    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")
    
    # 遍历文件夹
    for _, _, video_name in os.walk(folder_path):
        for video in video_name:
            # 检查文件扩展名
            if os.path.splitext(video)[1].lower() in video_extensions:
                # 获取文件的绝对路径
                
                video_files.append(video)
    
    # 按文件名排序（可选）
    video_files.sort()
    
    return video_files



def background_consistency(folder_path, clip_model, preprocess, video_list, device):
    sim = 0.0
    cnt = 0
    video_results = []
    image_transform = clip_transform(224)
    for video_path in tqdm(video_list, disable=get_rank() > 0):
        video_sim = 0.0
        cnt_per_video = 0
        images = load_video(os.path.join(folder_path, video_path))
        images = image_transform(images)
        images = images.to(device)
        image_features = clip_model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1, p=2)
        for i in range(len(image_features)):
            image_feature = image_features[i].unsqueeze(0)
            if i == 0:
                first_image_feature = image_feature
            else:
                sim_pre = max(0.0, F.cosine_similarity(former_image_feature, image_feature).item())
                sim_fir = max(0.0, F.cosine_similarity(first_image_feature, image_feature).item())
                cur_sim = (sim_pre + sim_fir) / 2
                video_sim += cur_sim
                cnt += 1
                cnt_per_video += 1
            former_image_feature = image_feature
        sim_per_image = video_sim / (len(image_features) - 1)
        sim += video_sim
        video_results.append({
            'video_path': video_path, 
            'video_results': sim_per_image,
            'video_sim': video_sim,
            'cnt_per_video': cnt_per_video})
    # sim_per_video = sim / (len(video_list) - 1)
    sim_per_frame = sim / cnt
    return sim_per_frame, video_results


def compute_background_consistency_file(folder_path, device, submodules_list, **kwargs):
    vit_path = submodules_list[0]
    clip_model, preprocess = clip.load(vit_path, device=device)
    
    video_list = get_video_files(folder_path)
    video_list = distribute_list_to_rank(video_list)
    all_results, video_results = background_consistency(folder_path, clip_model, preprocess, video_list, device)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        sim = sum([d['video_sim'] for d in video_results])
        cnt = sum([d['cnt_per_video'] for d in video_results])
        all_results = sim / cnt
    return all_results, video_results



def compute_background_consistency_video(video_path, device, submodules_list, **kwargs):
    vit_path = submodules_list[0]
    clip_model, preprocess = clip.load(vit_path, device=device)
    video_list = [video_path]
    all_results, video_results = background_consistency(clip_model, preprocess, video_list, device)
    return all_results, video_results




if __name__ == "__main__":
    CACHE_DIR = "/home/export/base/ycsc_chenkh/hitici_08/online1/Params/VBench"
    local = True
    submodules_dict = {}
    video_folder_path = "/home/export/base/ycsc_chenkh/hitici_08/online1/Datasets/video_data/vc_experiment/vc_result_opensora_"
    device = 'cuda'


    if local:
        vit_b_path = f'{CACHE_DIR}/clip_model/ViT-B-32.pt'
        if not os.path.isfile(vit_b_path):
            wget_command = ['wget', 'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt', '-P', os.path.dirname(vit_b_path)]
            subprocess.run(wget_command, check=True)
    else:
        vit_b_path = 'ViT-B/32'
    submodules_dict = [vit_b_path]

    all_results, video_results = compute_background_consistency_file(video_folder_path, device, submodules_dict)
    df = pd.DataFrame(video_results)
    df[['video_path', 'video_results']].to_csv("/home/export/base/ycsc_chenkh/hitici_08/online1/Files/VC_benchmark/file_back.csv",index=False)


