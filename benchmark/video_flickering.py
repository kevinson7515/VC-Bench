import os
import subprocess
import cv2
import numpy as np
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










def detect_flicker(folder_path,video_list, block_size=32, threshold_s=50):
    """
    检测视频中的闪烁现象
    
    参数:
        video_path (str): 视频文件路径
        block_size (int): 局部区域的大小（默认16x16像素）
        threshold_s (int): 闪烁判定阈值（差异绝对值之和超过此值则认为闪烁）
    
    返回:
        flicker_report (dict): 闪烁统计结果，包括总闪烁区域数和各帧的闪烁区域数
    """
    video_results = []

    for video_path in tqdm(video_list, disable=get_rank() > 0):
        images = load_video(os.path.join(folder_path, video_path),return_tensor=False)
        total_frames = len(images)
        width = images[0].shape[1]
        height = images[0].shape[0]

        prev_frame_yuv = None
        prev_frame_hsv = None
        flicker_cnt = 0
        block_cnt = 0

        for frame_idx in range(total_frames):
            frame = images[frame_idx]
            frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
            if frame_idx > 0:
            # 初始化当前帧的闪烁区域计数器       
                # 将帧划分为block_size x block_size的块
                for y in range(0, height, block_size):
                    for x in range(0, width, block_size):
                        # 获取当前块
                        block_end_y = min(y + block_size, height)
                        block_end_x = min(x + block_size, width)
                        block_cnt += 1
                        # 计算YUV差异 
                        block_prev_yuv = prev_frame_yuv[y:block_end_y, x:block_end_x]
                        block_curr_yuv = frame_yuv[y:block_end_y, x:block_end_x]
                        diff_yuv =   np.abs(block_curr_yuv.astype(np.int16) - block_prev_yuv.astype(np.int16)) / 255.0
                        diff_yuv = np.sum(diff_yuv) / (block_size*block_size*3)

                        # 计算HSV差异 (V分量)
                        rgb_max = np.max(frame[y:block_end_y, x:block_end_x])
                        block_prev_hsv = prev_frame_hsv[y:block_end_y, x:block_end_x]
                        block_curr_hsv = frame_hsv[y:block_end_y, x:block_end_x]
                        diff_hsv = np.abs(block_curr_hsv.astype(np.int16) - block_prev_hsv.astype(np.int16))
                        diff_hsv[:,:,0] =  diff_hsv[:,:,0] / 360.0
                        diff_hsv[:,:,1] =  diff_hsv[:,:,0] 
                        diff_hsv[:,:,2] =  diff_hsv[:,:,0] / rgb_max
                        diff_hsv = np.sum(diff_hsv) / (block_size*block_size*3)
                        
                        # 合并差异
                        total_diff = (diff_yuv + diff_hsv) / 2 
                        
                        # 判断是否闪烁
                        if total_diff > threshold_s:
                            flicker_cnt += 1
            # 更新前一帧
            prev_frame_yuv = frame_yuv.copy()
            prev_frame_hsv = frame_hsv.copy()

        
        video_results.append({'video_path': video_path, 'video_results': flicker_cnt/block_cnt})
    average_results = sum([o['video_results'] for o in video_results]) / len(video_results)
    return average_results, video_results
      



def compute_imaging_quality_file(folder_path):
    video_list = get_video_files(folder_path)
    video_list = distribute_list_to_rank(video_list)
    all_results, video_results = detect_flicker(folder_path, video_list,threshold_s=0.01)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    return all_results, video_results



def compute_imaging_quality_video(folder_path, device, submodules_list, **kwargs):
    pass






# 使用示例
if __name__ == "__main__":
    video_folder_path ="/home/export/base/ycsc_chenkh/hitici_08/online1/Datasets/video_data/vc_experiment/vc_result_opensora_"  # 替换为你的视频路径

    all_results, video_results =  compute_imaging_quality_file(video_folder_path) 
    df = pd.DataFrame(video_results)
    df[['video_path', 'video_results']].to_csv("/home/export/base/ycsc_chenkh/hitici_08/online1/Files/VC_benchmark/file_fli.csv",index=False)
    
    