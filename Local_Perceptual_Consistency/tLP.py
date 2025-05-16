import cv2
import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from pathlib import Path

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 手动加载本地 VGG16 模型
vgg16_path = "vgg16-397923af.pth"  # 替换为你的本地文件路径
vgg = models.vgg16().to(device)  # 创建 VGG16 模型（不加载预训练权重）

# 加载本地权重
state_dict = torch.load(vgg16_path, map_location=device)
vgg.load_state_dict(state_dict)

# 使用 VGG16 的前16层，并设置为评估模式
vgg = vgg.features[:16].to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False

# 图像预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. 定义函数：计算感知损失 (LP)
def compute_perceptual_loss(frame1, frame2):
    frame1 = transform(frame1).unsqueeze(0).to(device)
    frame2 = transform(frame2).unsqueeze(0).to(device)
    
    feat1 = vgg(frame1)
    feat2 = vgg(frame2)
    
    loss = torch.nn.functional.l1_loss(feat1, feat2).item()
    return loss

# 3. 定义函数：计算光流 (OF)
def compute_optical_flow(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow_map = flow.calc(gray1, gray2, None)
    
    return flow_map

# 4. 计算 tOF 和 tLP
def compute_tOF_tLP(generated_frames):
    num_frames = len(generated_frames)
    
    if num_frames < 2:
        raise ValueError("视频帧数少于 2 帧，无法计算 tOF 和 tLP！")
    
    tLP_values = []
    
    # 获取帧的尺寸（假设所有帧尺寸相同）
    height, width = generated_frames[0].shape[:2]  # 提取高度和宽度
    
    # 逐对帧计算
    for t in range(1, num_frames):
        gen_frame_t1, gen_frame_t = generated_frames[t-1], generated_frames[t]
        
        # 计算感知损失
        gen_lp = compute_perceptual_loss(gen_frame_t1, gen_frame_t)
        print(gen_lp)
        
        # 计算感知相似度
        tLP = abs(1 - gen_lp)
        tLP_values.append(tLP)
    
    # 取平均值
    avg_tLP = np.mean(tLP_values)
    
    return avg_tLP

# 5. 从视频中提取帧
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    return frames

def calculate_all_tOF_tLP_scores(true_video, model_dirs, output_csv):    
    # 获取 video_caption.csv 文件第一列的所有视频文件名(不包括后缀.mp4)
    true_videos = []
    with open(true_video, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row:  # 确保行不为空
                video_name = row[0]
                true_videos.append(video_name + '.mp4')  # 重新添加.mp4以便后续处理
    
    # 准备 CSV 文件
    with open(output_csv, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # 写入标题行（5列）
        csv_writer.writerow(['video_filename', 'OpenSora2.0_tLP_score'])

        for video_name in true_videos:
            row = [video_name]  # 第一列是视频文件名

            # 遍历每个模型的生成视频文件夹
            for model_dir in model_dirs:
                generated_path = os.path.join(model_dir, video_name)
                
                # 检查生成视频是否存在
                if not os.path.exists(generated_path):
                    print(f"Warning: Generated video {video_name} not found in {model_dir}, skipping...")
                    row.append("N/A")  # 如果不存在，填充 "N/A"
                    continue
                
                try:
                    generated_frames = extract_frames(generated_path)
                    avg_tLP = compute_tOF_tLP(generated_frames)
                    score = avg_tLP
                    print(f"Calculated tLP for {video_name} in {model_dir}: {score}")
                    row.append(score)
                except Exception as e:
                    print(f"Error processing {video_name} in {model_dir}: {str(e)}")
                    row.append("Error")  # 如果计算失败，填充 "Error"
                    continue
            
            # 写入 CSV 行
            csv_writer.writerow(row)

if __name__ == "__main__":
    # 设置目录路径
    true_video = "../True_video/video_caption.csv"
    # 四个模型的生成视频文件夹
    model_dirs = [
        "../Open-Sora2.0_result"
    ]
    output_csv_path = "../tLP_scores_opensora.csv"  # CSV 文件路径
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    # 计算所有 FVD 分数
    calculate_all_tOF_tLP_scores(true_video, model_dirs, output_csv_path)
    
    print(f"All tLP scores have been saved to {output_csv_path}")