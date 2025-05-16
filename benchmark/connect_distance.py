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
from benchmark.utils import video_to_tensor
from benchmark.distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)
import numpy as np
from skimage.metrics import structural_similarity as ssim
import random
from benchmark.dtw import *
from benchmark.utils import load_video, load_dimension_info




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




# 处理numpy数据
def calculate_ssim(ori_img, gen_img):
    """
    计算原视频与生成视频的结构相似性指数（SSIM）
    
    参数:
    original_video (numpy.ndarray): 原视频，形状为 (帧数, 高度, 宽度, 通道数)
    generated_video (numpy.ndarray): 生成视频，形状为 (帧数, 高度, 宽度, 通道数)
    
    返回:
    ssim_values (numpy.ndarray): 每帧的SSIM值，形状为 (帧数,)
    average_ssim (float): 所有帧的平均SSIM值
    """
    if ori_img.shape != gen_img.shape:
        raise ValueError("形状必须相同")
    
    if ori_img.ndim == 3:  # 彩色视频
        ssim_value = 0 
        for channel in range(3):         
            ssim_value += ssim(ori_img[:, :, channel], 
                                 gen_img[:, :, channel], 
                                 data_range=255)
        ssim_value = ssim_value / 3
    else:  # 灰度视频
        ssim_value = ssim(ori_img, gen_img, data_range=255)
    
    return ssim_value




# 传入首视频，尾视频，和整个生成视频
# start_DTW 首视频的对应关系    end_DTW 尾视频的对应关系   分别从[0，0]开始
def calculate_connect_distance(ori_video_start, ori_video_end, gen_video,  start_DTW_list, end_DTW_list, ori_num, gen_num):
    
    start_len = len(ori_video_start)
    end_len = len(ori_video_end)
    gen_len = len(gen_video) - start_len - end_len
    
    e_list = []
    for _ in range(gen_num):
        gen_id_1 = random.randint(start_len, start_len+9)
        gen_img_1 = gen_video[gen_id_1]

        gen_id_2 = random.randint(start_len + gen_len -10, start_len + gen_len-1)
        gen_img_2 = gen_video[gen_id_2]


        for _ in range(ori_num):
            id = random.randint(len(start_DTW_list)-10, len(start_DTW_list)-1)
            ori_start_id = start_DTW_list[id][0]
            gen_start_id = start_DTW_list[id][1]
            ori_start_img = ori_video_start[ori_start_id]
            gen_start_img = gen_video[gen_start_id]

            start_ssim_1 = calculate_ssim(ori_start_img, gen_img_1)
            start_ssim_2 = calculate_ssim(gen_start_img, gen_img_1)
           
            start_e = abs((1 - start_ssim_1)  / np.sqrt(gen_id_1 - ori_start_id) - (1 - start_ssim_2) / np.sqrt(gen_id_1 - gen_start_id))   
            #start_e = abs((1 - start_ssim_1)   - (1 - start_ssim_2) ) 

            id = random.randint(0, 9)
            ori_end_id = end_DTW_list[id][0]
            gen_end_id = end_DTW_list[id][1]
            ori_end_img = ori_video_end[ori_end_id]
            gen_end_img = gen_video[start_len + gen_len + gen_end_id]

            end_ssim_1 = calculate_ssim(ori_end_img, gen_img_2)
            end_ssim_2 = calculate_ssim(gen_end_img, gen_img_2)
            
            end_e = abs((1 - end_ssim_1)  / np.sqrt(ori_end_id + gen_len + start_len - gen_id_2) - (1 - end_ssim_2) / np.sqrt(gen_end_id + gen_len + start_len - gen_id_2))

            # end_e = abs((1 - end_ssim_1)   - (1 - end_ssim_2) )

            frame_e = (start_e + end_e) / 2
            e_list.append(frame_e)

    return e_list,  2.0 / (1.0 + np.exp(-np.mean(e_list)*10)) - 1.0




def evaluate_connect_distance(gt_folder_path, gen_folder_path, video_list):
    video_results = []
    for video_path in tqdm(video_list, disable=get_rank() > 0):
        
       # opensora_video_path = os.path.splitext(video_path)[0]+'prompt_0000.mp4'

    
        gt_video = load_video(os.path.join(gt_folder_path, video_path),return_tensor=False)
        gen_video = load_video(os.path.join(gen_folder_path, video_path),return_tensor=False)

        if gt_video[0].shape[0] != gen_video[0].shape[0]:
            video_results.append({'video_path': video_path, 'video_results': 0})

        else:
            start_video = gt_video[:24]
            end_video = gt_video[-24:]
        
            start_len = len(start_video)
            end_len = len(end_video)
        
            start_DTW_list = dtw(video_to_features(start_video), video_to_features(gen_video[:start_len]))
            end_DTW_list = dtw(video_to_features(end_video), video_to_features(gen_video[-end_len:]))
    
            ssim_list, avg = calculate_connect_distance(start_video, end_video, gen_video, start_DTW_list, end_DTW_list, 5, 5)
            video_results.append({'video_path': video_path, 'video_results': avg})
    average_results = sum([o['video_results'] for o in video_results]) / len(video_results)
    return average_results, video_results






def compute_imaging_quality_file(gt_folder_path,gen_folder_path):
    video_list = get_video_files(gen_folder_path)
    video_list = distribute_list_to_rank(video_list)
    
    all_results, video_results = evaluate_connect_distance(gt_folder_path, gen_folder_path, video_list)

    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    return all_results, video_results






gen_folder_path = "/home/export/base/ycsc_chenkh/hitici_08/online1/Datasets/video_data/vc_experiment/vc_result_opensora_"
gt_folder_path = "/home/export/base/ycsc_chenkh/hitici_08/online1/Datasets/video_data/vc_experiment/video_400"
avg, video_results = compute_imaging_quality_file(gt_folder_path, gen_folder_path)
df = pd.DataFrame(video_results)
df[['video_path', 'video_results']].to_csv("/home/export/base/ycsc_chenkh/hitici_08/online1/Files/VC_benchmark/file_cnt.csv",index=False)




# 0 opensora  1  wan  2 cogvideox






