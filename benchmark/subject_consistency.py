import io
import os
import subprocess
import pandas as pd
import cv2
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys
current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from benchmark.utils import load_video,  dino_transform
import logging

from benchmark.distributed import (
    get_world_size,                             # 获取分布式训练中的总进程数
    get_rank,                                   # 获取当前进程的排名（0为主进程）
    all_gather,                                 # 聚合所有进程的数据
    barrier,                                    # 同步所有进程的屏障
    distribute_list_to_rank,                    # 将数据列表分发给各进程
    gather_list_of_dict,                        # 聚合所有进程的字典列表
)

# 设置日志格式，记录时间、模块名、日志级别和消息
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)





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





def subject_consistency(folder_path, model, video_list, device): 
    """
    model:用于提取图像特征的模型(如DINO)。

    video_list:待处理的视频路径列表。

    device:计算设备(如cuda:0)。
    
    """
    # 累积相似度总和
    sim = 0.0
    # 累积帧对数
    cnt = 0
    # 存储每个视频的结果
    video_results = []
    image_transform = dino_transform(224)

    # 帧读取逻辑
    for video_path in tqdm(video_list, disable=get_rank() > 0):
        video_sim = 0.0
        # 如果是帧文件夹
        # if read_frame:
        #     video_path = video_path[:-4].replace('videos', 'frames').replace(' ', '_')
        #     tmp_paths = [os.path.join(video_path, f) for f in sorted(os.listdir(video_path))]
        #     images = []
        #     for tmp_path in tmp_paths:
        #         images.append(image_transform(Image.open(tmp_path)))
        # # 直接读取视频
        # else:
        images = load_video(os.path.join(folder_path, video_path))
        images = image_transform(images)

        # 帧特征提取与相似度计算
        for i in range(len(images)):
            with torch.no_grad():
                image = images[i].unsqueeze(0)
                image = image.to(device)
                image_features = model(image)
                image_features = F.normalize(image_features, dim=-1, p=2)
                if i == 0:
                    first_image_features = image_features
                else:
                    # 计算当前帧与前帧的相似度
                    sim_pre = max(0.0, F.cosine_similarity(former_image_features, image_features).item())
                    # 计算当前帧与第一帧的相似度
                    sim_fir = max(0.0, F.cosine_similarity(first_image_features, image_features).item())
                    #  # 平均相似度
                    cur_sim = (sim_pre + sim_fir) / 2
                    video_sim += cur_sim
                    cnt += 1
            former_image_features = image_features
        # 某个视频的帧间相似度
        sim_per_images = video_sim / (len(images) - 1)
        sim += video_sim
        video_results.append({'video_path': video_path, 'video_results': sim_per_images})
    # sim_per_video = sim / (len(video_list) - 1)
    # 全局平均相似度
    sim_per_frame = sim / cnt
    return sim_per_frame, video_results


def compute_subject_consistency_file(video_folder_path, device, submodules_list, **kwargs):
    '''
    json_dir:包含视频路径和标注的JSON文件目录。

    submodules_list:模型加载参数(如DINO的配置)。
    '''

    dino_model = torch.hub.load(**submodules_list).to(device)
    logger.info("Initialize DINO success")
    video_list = get_video_files(video_folder_path)
    video_list = distribute_list_to_rank(video_list)
    all_results, video_results = subject_consistency(video_folder_path, dino_model, video_list, device)   
    if get_world_size() > 1: 
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    return all_results, video_results




def compute_subject_consistency_video(video_path, device, submodules_list, **kwargs):
    '''
    json_dir:包含视频路径和标注的JSON文件目录。

    submodules_list:模型加载参数(如DINO的配置)。
    '''
    dino_model = torch.hub.load(**submodules_list).to(device)
    logger.info("Initialize DINO success")
    video_list = [video_path]
    all_results, video_results = subject_consistency(dino_model, video_list, device)    
    
    return all_results, video_results



if __name__ == "__main__":
    CACHE_DIR = "/home/export/base/ycsc_chenkh/hitici_08/online1/Params/VBench"
    local = True
    submodules_dict = {}
    video_folder_path = "/home/export/base/ycsc_chenkh/hitici_08/online1/Datasets/video_data/vc_experiment/vc_result_opensora_"
    device = 'cuda'

    if local:
        submodules_dict = {
            'repo_or_dir': f'{CACHE_DIR}/dino_model/facebookresearch_dino_main',
            'path': f'{CACHE_DIR}/dino_model/dino_vitbase16_pretrain.pth', 
            'model': 'dino_vitb16',
            'source': 'local',
                    }
                
        details = submodules_dict
        # Check if the file exists, if not, download it with wget
        if not os.path.isdir(details['repo_or_dir']):
            print(f"Directory {details['repo_or_dir']} does not exist. Cloning repository...") 
            subprocess.run(['git', 'clone', 'https://github.com/facebookresearch/dino', details['repo_or_dir']], check=True)

        if not os.path.isfile(details['path']):
            print(f"File {details['path']} does not exist. Downloading...")  
            wget_command = ['wget', '-P', os.path.dirname(details['path']),
                            'https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth']
            subprocess.run(wget_command, check=True)
                
                
    else:
        submodules_dict = {
            'repo_or_dir':'facebookresearch/dino:main',
            'source':'github',
            'model': 'dino_vitb16',
                    }

    all_results, video_results = compute_subject_consistency_file(video_folder_path, device, submodules_dict)
    df = pd.DataFrame(video_results)
    df[['video_path', 'video_results']].to_csv("/home/export/base/ycsc_chenkh/hitici_08/online1/Files/VC_benchmark/file_sub.csv",index=False)