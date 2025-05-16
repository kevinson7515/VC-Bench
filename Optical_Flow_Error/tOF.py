import cv2
import os
import csv
import numpy as np
from collections import defaultdict

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

def compute_optical_flow(frame1, frame2):
    """计算两帧之间的光流"""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    patch_size = flow.getPatchSize()
    print(f"DISOpticalFlow PRESET_MEDIUM 窗口大小: {patch_size}x{patch_size}")
    
    flow_map = flow.calc(gray1, gray2, None)
    
    return flow_map

def compute_tOF(generated_frames, real_frames):
    """计算tOF指标"""
    num_frames = min(len(generated_frames), len(real_frames))
    
    if num_frames < 2:
        print("警告：视频帧数少于2帧，无法计算tOF！")
        return None
    
    tOF_values = []
    
    # 获取帧的尺寸（假设所有帧尺寸相同）
    height, width = generated_frames[0].shape[:2]  # 提取高度和宽度
    
    # 逐对帧计算
    for t in range(1, num_frames):
        try:
            gen_frame_t1, gen_frame_t = generated_frames[t-1], generated_frames[t]
            real_frame_t1, real_frame_t = real_frames[t-1], real_frames[t]
            
            # 计算光流
            gen_flow = compute_optical_flow(gen_frame_t1, gen_frame_t)
            real_flow = compute_optical_flow(real_frame_t1, real_frame_t)
            
            # 计算 tOF（光流的 L1 差，归一化到像素总数）
            tOF = np.abs(gen_flow - real_flow).sum() / (height * width)  # 除以像素总数
            tOF_values.append(tOF)
        except Exception as e:
            print(f"计算第{t}帧光流时出错: {e}")
            continue
       
    # 取平均值
    if tOF_values:
        avg_tOF = np.mean(tOF_values)
        return avg_tOF
    else:
        return None

def calculate_head_tail_tOF(true_head_path, true_tail_path, generated_path):
    """计算头尾视频与生成视频对应部分的tOF"""
    try:
        # 提取所有视频帧
        head_frames = extract_frames(true_head_path)
        tail_frames = extract_frames(true_tail_path)
        generated_frames = extract_frames(generated_path)
        
        if not head_frames or not tail_frames or not generated_frames or len(head_frames) < 2 or len(tail_frames) < 2 or len(generated_frames) < 2:
            print(f"警告: 视频帧提取为空或帧数过少 - {true_head_path}, {true_tail_path}, 或 {generated_path}")
            return None
        
        # 计算头部视频的tOF（与生成视频的开头部分比较）
        head_gen_frames = generated_frames[:len(head_frames)]
        head_tOF = compute_tOF(head_gen_frames, head_frames)
        
        # 计算尾部视频的tOF（与生成视频的结尾部分比较）
        tail_gen_frames = generated_frames[-len(tail_frames):]
        tail_tOF = compute_tOF(tail_gen_frames, tail_frames)
        
        # 计算平均tOF
        if head_tOF is not None and tail_tOF is not None:
            avg_tOF = (head_tOF + tail_tOF) / 2
            return avg_tOF
        elif head_tOF is not None:
            return head_tOF
        elif tail_tOF is not None:
            return tail_tOF
        else:
            return None
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
    csv_file = "../Optical_Flow_Consistency_opensora.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Video_ID", "tOF"])
        
        # 遍历每个分组，计算tOF
        for xxx, files in video_groups.items():
            if "head" in files and "tail" in files:
                # 检查Generated_video中是否存在对应的XXX.mp4
                generated_path = os.path.join(generated_video_dir, f"{xxx}.mp4")
                if os.path.exists(generated_path):
                    print(f"处理视频: {xxx}")
                    tof_value = calculate_head_tail_tOF(
                        files["head"], 
                        files["tail"], 
                        generated_path
                    )
                    
                    if tof_value is not None:
                        writer.writerow([xxx, tof_value])
                        print(f"  tOF: {tof_value:.4f}")
                    else:
                        print(f"  无法计算tOF")
                else:
                    print(f"警告: 在Generated_video中未找到对应的视频: {xxx}.mp4")
    
    print(f"结果已保存到 {csv_file}")

if __name__ == "__main__":
    main()