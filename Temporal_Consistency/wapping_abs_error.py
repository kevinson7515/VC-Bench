# 值越小，光滑度越好
import cv2
import numpy as np
import os
import csv
from tqdm import tqdm

# 计算光流（使用 OpenCV 的 Farneback 方法）
def compute_optical_flow(frame1, frame2):
    # 转换为灰度图
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    # 计算稠密光流
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

# Warping 函数
def warp_frame(frame, flow):
    h, w = flow.shape[:2]
    map_x = np.tile(np.arange(w), (h, 1)).astype(np.float32)
    map_y = np.repeat(np.arange(h)[:, None], w, axis=1).astype(np.float32)
    map_x += flow[:, :, 0]
    map_y += flow[:, :, 1]
    warped = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
    return warped

# 计算基于绝对差异的Warping Error（RGB三通道平均）
def compute_warping_error(original, warped):
    # 计算绝对差异
    abs_diff = cv2.absdiff(original, warped)
    
    # 转换为浮点型以计算平均值
    abs_diff = abs_diff.astype(np.float32)
    
    # 计算每个通道的平均绝对差异
    mean_diff_per_channel = np.mean(abs_diff, axis=(0, 1))
    
    # 取三通道的平均值
    mean_diff = np.mean(mean_diff_per_channel)
    
    return mean_diff

# 主函数
def calculate_smoothness_score(video_path):
    # 读取视频
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return None

    errors = []
    ret, prev_frame = video.read()
    if not ret:
        print(f"无法读取视频第一帧: {video_path}")
        return None

    frame_count = 0
    while video.isOpened():
        ret, curr_frame = video.read()
        if not ret:
            break
        frame_count += 1

        # 计算光流
        flow = compute_optical_flow(prev_frame, curr_frame)

        # Warping
        warped_frame = warp_frame(prev_frame, flow)

        # 计算基于绝对差异的Warping Error
        error = compute_warping_error(curr_frame, warped_frame)
        errors.append(error)

        prev_frame = curr_frame

    video.release()

    # 计算平均 Warping Error
    if errors:
        avg_error = np.mean(errors)
    else:
        print(f"未计算到任何 Warping Error: {video_path}")
        return None

    return avg_error

def process_videos_in_folder(folder_path, output_csv):
    # 支持的视频文件扩展名
    video_extensions = ('.mp4')
    
    # 获取文件夹中的所有视频文件
    video_files = [f for f in os.listdir(folder_path) 
                  if f.lower().endswith(video_extensions)]
    
    if not video_files:
        print(f"文件夹中没有找到视频文件: {folder_path}")
        return
    
    # 创建CSV文件并写入头
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Video File', 'Smoothness Score'])
        
        # 处理每个视频文件
        for video_file in tqdm(video_files, desc="Processing videos"):
            video_path = os.path.join(folder_path, video_file)
            score = calculate_smoothness_score(video_path)
            
            if score is not None:
                csv_writer.writerow([video_file, score])
                print(f"Processed: {video_file} - Score: {score:.4f}")
            else:
                print(f"Skipped: {video_file} - Error in processing")

def calculate_all_wapping_errors(true_video, model_dirs, output_csv):    
    # 获取 video_caption.csv 文件第一列的所有视频文件名(不包括后缀.mp4)
    true_videos = []
    with open(true_video, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row:  # 确保行不为空
                video_name = row[0].split('.')[0]  # 去掉.mp4后缀
                true_videos.append(video_name + '.mp4')  # 重新添加.mp4以便后续处理
    
    # 准备 CSV 文件
    with open(output_csv, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # 写入标题行（5列）
        csv_writer.writerow(['video_filename', 'opensora_wapping_error'])

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
                    score = calculate_smoothness_score(generated_path)
                    score = 1 - score / 255
                    print(f"Calculated wapping_error for {video_name} in {model_dir}: {score}")
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
        "../Open-Sora2.0_result",
    ]
    output_csv_path = "../Temporal_consistency_opensora.csv"  # CSV 文件路径
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    calculate_all_wapping_errors(true_video, model_dirs, output_csv_path)
    
    print(f"All wapping errors have been saved to {output_csv_path}")