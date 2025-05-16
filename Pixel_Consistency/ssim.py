import cv2
import numpy as np
import os
import csv
import re
from skimage.metrics import structural_similarity as ssim
from collections import defaultdict

def calculate_ssim_rgb(frame1, frame2):
    """计算两帧之间的RGB-SSIM"""
    frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    
    ssim_channels = []
    for channel in range(3):  # 遍历 R、G、B 通道
        ch1 = frame1_rgb[:, :, channel]
        ch2 = frame2_rgb[:, :, channel]
        score = ssim(ch1, ch2, data_range=ch1.max() - ch1.min())
        ssim_channels.append(score)
    
    return np.mean(ssim_channels)

def extract_frames(video_path):
    """提取视频的所有帧"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames

def calculate_head_tail_ssim(true_head_path, true_tail_path, generated_path):
    """计算头尾视频与生成视频对应部分的SSIM"""
    # 提取所有视频帧
    try:
        head_frames = extract_frames(true_head_path)
        tail_frames = extract_frames(true_tail_path)
        generated_frames = extract_frames(generated_path)
        
        if not head_frames or not tail_frames or not generated_frames:
            print(f"警告: 视频帧提取为空 - {true_head_path}, {true_tail_path}, 或 {generated_path}")
            return None
        
        # 计算头部视频的SSIM（与生成视频的开头部分比较）
        head_ssim_scores = []
        for i in range(min(len(head_frames), len(generated_frames))):
            score = calculate_ssim_rgb(head_frames[i], generated_frames[i])
            head_ssim_scores.append(score)
        
        # 计算尾部视频的SSIM（与生成视频的结尾部分比较）
        tail_ssim_scores = []
        for i in range(min(len(tail_frames), len(generated_frames))):
            gen_idx = len(generated_frames) - min(len(tail_frames), len(generated_frames)) + i
            score = calculate_ssim_rgb(tail_frames[i], generated_frames[gen_idx])
            tail_ssim_scores.append(score)
        
        # 计算平均SSIM
        avg_head_ssim = np.mean(head_ssim_scores) if head_ssim_scores else 0
        avg_tail_ssim = np.mean(tail_ssim_scores) if tail_ssim_scores else 0
        avg_ssim = (avg_head_ssim + avg_tail_ssim) / 2
        
        return avg_ssim
    except Exception as e:
        print(f"处理视频时出错: {e} - {true_head_path}, {true_tail_path}, {generated_path}")
        return None

def main():
    # 设置文件夹路径
    true_video_dir = "../True_video"  # 请替换为您的True_video文件夹的实际路径
    generated_video_dir = "../Open-Sora2.0_result"  # 请替换为您的Generated_video文件夹的实际路径
    
    # 确保文件夹存在
    if not os.path.exists(true_video_dir):
        raise ValueError(f"文件夹不存在: {true_video_dir}")
    if not os.path.exists(generated_video_dir):
        raise ValueError(f"文件夹不存在: {generated_video_dir}")
    
    # 获取True_video文件夹中的所有视频文件
    true_video_files = [f for f in os.listdir(true_video_dir) if f.endswith('.mp4')]
    
    # 按XXX部分分组
    video_groups = defaultdict(dict)
    for video_file in true_video_files:
        if "_head.mp4" in video_file:
            xxx = video_file.replace("_head.mp4", "")
            video_groups[xxx]["head"] = os.path.join(true_video_dir, video_file)
        elif "_tail.mp4" in video_file:
            xxx = video_file.replace("_tail.mp4", "")
            video_groups[xxx]["tail"] = os.path.join(true_video_dir, video_file)
    
    # 准备CSV文件
    csv_file = "../Pixel_Consistency_opensora.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Video_ID", "SSIM"])
        
        # 遍历每个分组，计算SSIM
        for xxx, files in video_groups.items():
            if "head" in files and "tail" in files:
                # 检查Generated_video中是否存在对应的XXX.mp4
                generated_path = os.path.join(generated_video_dir, f"{xxx}.mp4")
                if os.path.exists(generated_path):
                    print(f"处理视频: {xxx}")
                    ssim_value = calculate_head_tail_ssim(
                        files["head"], 
                        files["tail"], 
                        generated_path
                    )
                    
                    if ssim_value is not None:
                        writer.writerow([xxx, ssim_value])
                        print(f"  SSIM: {ssim_value:.4f}")
                    else:
                        print(f"  无法计算SSIM")
                else:
                    print(f"警告: 在Generated_video中未找到对应的视频: {xxx}.mp4")
    
    print(f"结果已保存到 {csv_file}")

if __name__ == "__main__":
    main()