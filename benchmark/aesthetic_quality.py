import os
import clip
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess
from urllib.request import urlretrieve
from tqdm import tqdm
import sys
current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None
from benchmark.utils import load_video, load_dimension_info, clip_transform
from benchmark.distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)

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








# 加载LAION美学评分模型（线性回归模型，基于CLIP-ViT-L/14特征）
def get_aesthetic_model(cache_folder):
    """load the aethetic model"""
    path_to_model = cache_folder + "/sa_0_4_vit_l_14_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true"
        )
        # download aesthetic predictor
        if not os.path.isfile(path_to_model):
            try:
                print(f'trying urlretrieve to download {url_model} to {path_to_model}')
                urlretrieve(url_model, path_to_model) # unable to download https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true to pretrained/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth 
            except:
                print(f'unable to download {url_model} to {path_to_model} using urlretrieve, trying wget')
                wget_command = ['wget', url_model, '-P', os.path.dirname(path_to_model)]
                subprocess.run(wget_command)
    # 定义模型结构（768维输入 -> 1维输出）
    m = nn.Linear(768, 1)
    # # 加载预训练权重
    s = torch.load(path_to_model)
    # # 加载权重
    m.load_state_dict(s)
    m.eval()
    return m


# 计算视频的美学质量评分（基于LAION美学模型）
def laion_aesthetic(folder_path,aesthetic_model, clip_model, video_list, device):
    batch_size = 32
    aesthetic_model.eval()
    clip_model.eval()
    aesthetic_avg = 0.0          # 全局平均分
    num = 0
    video_results = []

    for video_path in tqdm(video_list, disable=get_rank() > 0):
        images = load_video(os.path.join(folder_path, video_path))
        image_transform = clip_transform(224)

        aesthetic_scores_list = []
        for i in range(0, len(images), batch_size):
            # 按batch处理帧（避免显存溢出）
            image_batch = images[i:i + batch_size]
            image_batch = image_transform(image_batch)
            image_batch = image_batch.to(device)

            with torch.no_grad():
                # 提取CLIP特征并归一化
                image_feats = clip_model.encode_image(image_batch).to(torch.float32)
                image_feats = F.normalize(image_feats, dim=-1, p=2)
                # 预测美学评分
                aesthetic_scores = aesthetic_model(image_feats).squeeze(dim=-1)

            aesthetic_scores_list.append(aesthetic_scores)
        
         # 合并所有batch的评分
        aesthetic_scores = torch.cat(aesthetic_scores_list, dim=0)
        normalized_aesthetic_scores = aesthetic_scores / 10              # 归一化到0-10分制
        cur_avg = torch.mean(normalized_aesthetic_scores, dim=0, keepdim=True)
        aesthetic_avg += cur_avg.item()          # 更新全局平均
        num += 1
        video_results.append({'video_path': video_path, 'video_results': cur_avg.item()})

    aesthetic_avg /= num
    return aesthetic_avg, video_results


def compute_aesthetic_quality_file(folder_path, device, submodules_list, **kwargs):
    vit_path = submodules_list[0]
    aes_path = submodules_list[1]
    # 多GPU训练时，主进程下载模型，其他进程等待
    if get_rank() == 0:
        aesthetic_model = get_aesthetic_model(aes_path).to(device)     
        barrier()       # 主进程下载完成后，释放屏障
    else:
        barrier()       # 其他进程在此等待
        aesthetic_model = get_aesthetic_model(aes_path).to(device)     # 直接加载已下载的模型

    # 加载CLIP模型
    clip_model, preprocess = clip.load(vit_path, device=device)
    # 从JSON加载视频列表并分配各进程
    
    video_list = get_video_files(folder_path)
    video_list = distribute_list_to_rank(video_list)
    # 计算美学评分
    all_results, video_results = laion_aesthetic(folder_path, aesthetic_model, clip_model, video_list, device)
    # 分布式聚合结果
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    return all_results, video_results




def compute_aesthetic_quality_video(video_path, device, submodules_list, **kwargs):
    vit_path = submodules_list[0]
    aes_path = submodules_list[1]
    aesthetic_model = get_aesthetic_model(aes_path).to(device)     
      
    # 加载CLIP模型
    clip_model, preprocess = clip.load(vit_path, device=device)
    video_list = [video_path]
   
    # 计算美学评分
    all_results, video_results = laion_aesthetic(aesthetic_model, clip_model, video_list, device)
    return all_results, video_results







if __name__ == "__main__":
    CACHE_DIR = "/home/export/base/ycsc_chenkh/hitici_08/online1/Params/VBench"
    local = True
    submodules_dict = {}
    video_folder_path = "/home/export/base/ycsc_chenkh/hitici_08/online1/Datasets/video_data/vc_experiment/vc_result_opensora_"
    device = 'cuda'

    aes_path = f'{CACHE_DIR}/aesthetic_model/emb_reader'
    if local:
        vit_l_path = f'{CACHE_DIR}/clip_model/ViT-L-14.pt'
        if not os.path.isfile(vit_l_path):
            wget_command = ['wget' ,'https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt', '-P', os.path.dirname(vit_l_path)]
            subprocess.run(wget_command, check=True)
    else:
        vit_l_path = 'ViT-L/14'
    submodules_dict = [vit_l_path, aes_path]


    all_results, video_results = compute_aesthetic_quality_file(video_folder_path, device, submodules_dict)
    df = pd.DataFrame(video_results)
    df[['video_path', 'video_results']].to_csv("/home/export/base/ycsc_chenkh/hitici_08/online1/Files/VC_benchmark/file_aes.csv",index=False)




    
    
