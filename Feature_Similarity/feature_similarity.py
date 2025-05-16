import cv2
import numpy as np
import os
import csv
from scipy.spatial.distance import cdist
from collections import defaultdict
import gc

def normalize_chamfer_distance(A, B, image_shape):
    if len(A) == 0 or len(B) == 0:
        return 1.0
    
    min_A_to_B = np.min(cdist(A, B, 'euclidean'), axis=1)
    min_B_to_A = np.min(cdist(B, A, 'euclidean'), axis=0)
    
    chamfer_dist = (np.mean(min_A_to_B) + np.mean(min_B_to_A)) / 2
    max_possible_dist = np.sqrt(image_shape[0]**2 + image_shape[1]**2)
    normalized_dist = chamfer_dist / max_possible_dist
    
    return np.clip(normalized_dist, 0.0, 1.0)

def extract_edge_points(frame, max_points=1000):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    points = np.column_stack(np.where(edges > 0))
    points = points[:, ::-1]
    
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
    
    return points

def extract_frames(video_path, resize_factor=0.5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 16 == 0:
            frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
            yield frame
        frame_count += 1
    
    cap.release()

def calculate_head_tail_chamfer(true_head_path, true_tail_path, generated_path):
    try:
        cap = cv2.VideoCapture(true_head_path)
        ret, first_frame = cap.read()
        cap.release()
        if not ret:
            print(f"无法读取视频第一帧: {true_head_path}")
            return None
        H, W = first_frame.shape[:2]
        
        head_chamfer_scores = []
        tail_chamfer_scores = []
        
        generated_points = [extract_edge_points(frame) for frame in extract_frames(generated_path)]
        if not generated_points:
            print(f"警告: 生成视频帧提取为空 - {generated_path}")
            return None
        
        for i, head_frame in enumerate(extract_frames(true_head_path)):
            if i >= len(generated_points):
                break
            head_point = extract_edge_points(head_frame)
            score = normalize_chamfer_distance(head_point, generated_points[i], (H, W))
            if np.isfinite(score):
                head_chamfer_scores.append(score)
            del head_point
            gc.collect()
        
        for i, tail_frame in enumerate(extract_frames(true_tail_path)):
            gen_idx = len(generated_points) - min(len(generated_points), i + 1)
            if gen_idx < 0:
                break
            tail_point = extract_edge_points(tail_frame)
            score = normalize_chamfer_distance(tail_point, generated_points[gen_idx], (H, W))
            if np.isfinite(score):
                tail_chamfer_scores.append(score)
            del tail_point
            gc.collect()
        
        del generated_points
        gc.collect()
        
        avg_head_chamfer = np.mean(head_chamfer_scores) if head_chamfer_scores else 1.0
        avg_tail_chamfer = np.mean(tail_chamfer_scores) if tail_chamfer_scores else 1.0
        avg_chamfer = (avg_head_chamfer + avg_tail_chamfer) / 2
        
        return 1 - avg_chamfer
    except Exception as e:
        print(f"处理视频时出错: {e} - {true_head_path}, {true_tail_path}, {generated_path}")
        return None

def main():
    true_video_dir = "../True_video"
    generated_video_dir = "../cogVideo2B_result"
    
    if not os.path.exists(true_video_dir):
        raise ValueError(f"文件夹不存在: {true_video_dir}")
    if not os.path.exists(generated_video_dir):
        raise ValueError(f"文件夹不存在: {generated_video_dir}")
    
    true_video_files = [f for f in os.listdir(true_video_dir) if f.endswith('.mp4')]
    
    video_groups = defaultdict(dict)
    for video_file in true_video_files:
        if "_head.mp4" in video_file:
            xxx = video_file.replace("_head.mp4", "")
            video_groups[xxx]["head"] = os.path.join(true_video_dir, video_file)
        elif "_tail.mp4" in video_file:
            xxx = video_file.replace("_tail.mp4", "")
            video_groups[xxx]["tail"] = os.path.join(true_video_dir, video_file)
    
    csv_file = "../Distribution_Similarity_cogvideox.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Video_ID", "CVD_score"])
        
        for xxx, files in video_groups.items():
            if "head" in files and "tail" in files:
                generated_path = os.path.join(generated_video_dir, f"{xxx}.mp4")
                if os.path.exists(generated_path):
                    print(f"处理视频: {xxx}")
                    chamfer_value = calculate_head_tail_chamfer(
                        files["head"], 
                        files["tail"], 
                        generated_path
                    )
                    
                    if chamfer_value is not None and np.isfinite(chamfer_value):
                        writer.writerow([xxx, chamfer_value])
                        print(f"  Normalized Chamfer Distance: {chamfer_value:.4f}")
                    else:
                        print(f"  无法计算Chamfer距离")
                else:
                    print(f"警告: 在Generated_video中未找到对应的视频: {xxx}.mp4")
    
    print(f"结果已保存到 {csv_file}")

if __name__ == "__main__":
    main()