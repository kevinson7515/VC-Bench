import os
import subprocess
import pandas as pd
import torch
from tqdm import tqdm
from torchvision import transforms
from pyiqa.archs.musiq_arch import MUSIQ
import sys
current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None
from benchmark.utils import load_video, load_dimension_info
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

def transform(images, preprocess_mode='shorter'):
    if preprocess_mode.startswith('shorter'):
        _, _, h, w = images.size()
        if min(h,w) > 512:
            scale = 512./min(h,w)
            images = transforms.Resize(size=( int(scale * h), int(scale * w) ), antialias=False)(images)
            if preprocess_mode == 'shorter_centercrop':
                images = transforms.CenterCrop(512)(images)

    elif preprocess_mode == 'longer':
        _, _, h, w = images.size()
        if max(h,w) > 512:
            scale = 512./max(h,w)
            images = transforms.Resize(size=( int(scale * h), int(scale * w) ), antialias=False)(images)

    elif preprocess_mode == 'None':
        return images / 255.

    else:
        raise ValueError("Please recheck imaging_quality_mode")
    return images / 255.

def technical_quality(folder_path, model, video_list, device, **kwargs):
    if 'imaging_quality_preprocessing_mode' not in kwargs:
        preprocess_mode = 'longer'
    else:
        preprocess_mode = kwargs['imaging_quality_preprocessing_mode']
    video_results = []
    for video_path in tqdm(video_list, disable=get_rank() > 0):
        images = load_video(os.path.join(folder_path, video_path))
        images = transform(images, preprocess_mode)
        acc_score_video = 0.
        for i in range(len(images)):
            frame = images[i].unsqueeze(0).to(device)
            # 逐帧进行质量评估
            score = model(frame)
            acc_score_video += float(score)
        video_results.append({'video_path': video_path, 'video_results': acc_score_video/len(images)})
    average_score = sum([o['video_results'] for o in video_results]) / len(video_results)
    average_score = average_score / 100.
    return average_score, video_results


def compute_imaging_quality_file(folder_path, device, submodules_list, **kwargs):
    model_path = submodules_list['model_path']

    model = MUSIQ(pretrained_model_path=model_path)
    model.to(device)
    model.training = False
    video_list = get_video_files(folder_path)
    video_list = distribute_list_to_rank(video_list)
    all_results, video_results = technical_quality(folder_path,model, video_list, device, **kwargs)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
        all_results = all_results / 100.
    return all_results, video_results


def compute_imaging_quality_video(video_path, device, submodules_list, **kwargs):
    model_path = submodules_list['model_path']

    model = MUSIQ(pretrained_model_path=model_path)
    model.to(device)
    model.training = False
    
    video_list = [video_path]
    all_results, video_results = technical_quality(model, video_list, device, **kwargs)
    return all_results, video_results






if __name__ == "__main__":
    CACHE_DIR = "/home/export/base/ycsc_chenkh/hitici_08/online1/Params/VBench"
    local = True
    submodules_dict = {}
    video_folder_path = "/home/export/base/ycsc_chenkh/hitici_08/online1/Datasets/video_data/vc_experiment/vc_result_opensora_"
    device = 'cuda'


    musiq_spaq_path = f'{CACHE_DIR}/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth'
    if not os.path.isfile(musiq_spaq_path):
        wget_command = ['wget', 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth', '-P', os.path.dirname(musiq_spaq_path)]
        subprocess.run(wget_command, check=True)
    submodules_dict = {'model_path': musiq_spaq_path}

    all_results, video_results = compute_imaging_quality_file(video_folder_path, device, submodules_dict)
    df = pd.DataFrame(video_results)
    df[['video_path', 'video_results']].to_csv("/home/export/base/ycsc_chenkh/hitici_08/online1/Files/VC_benchmark/file_img.csv",index=False)

