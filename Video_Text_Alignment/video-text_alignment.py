import torch
import clip
import csv
import cv2
import numpy as np
from PIL import Image
import os
from tqdm import tqdm


def load_clip_model():
    """
    加载CLIP模型，用于提取视频帧的视觉特征和文本的语义特征
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def extract_frames(video_path, num_frames=8):
    """
    从视频中等间隔提取指定数量的帧
    
    参数:
        video_path: 视频文件路径
        num_frames: 要提取的帧数
        
    返回:
        frames: 提取的帧列表
    """
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        return []
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return []
    
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        print(f"视频帧数为0: {video_path}")
        return []
    
    # 计算采样间隔
    sample_interval = max(1, total_frames // num_frames)
    
    for i in range(0, total_frames, sample_interval):
        if len(frames) >= num_frames:
            break
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        
        if ret:
            # 将BGR转换为RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
    
    cap.release()
    return frames

def compute_text_compliance(model, preprocess, device, video_path, text):
    """
    计算视频与文本描述的语义一致性（文本遵循度）
    
    参数:
        model: CLIP模型
        preprocess: CLIP预处理函数
        device: 设备（CPU或GPU）
        video_path: 视频文件路径
        text: 文本描述
        
    返回:
        score: 文本遵循度分数（0-1之间）
    """
    frames = extract_frames(video_path)
    
    if not frames:
        print(f"无法从视频中提取帧: {video_path}")
        return 0.0
    
    # 预处理帧并提取视觉特征
    processed_frames = []
    for frame in frames:
        pil_image = Image.fromarray(frame)
        processed_frame = preprocess(pil_image)
        processed_frames.append(processed_frame)
    
    # 将所有帧的张量堆叠在一起
    frame_tensor = torch.stack(processed_frames).to(device)
    
    # 预处理文本并提取文本特征
    text_token = clip.tokenize([text]).to(device)
    
    with torch.no_grad():
        # 提取图像特征
        image_features = model.encode_image(frame_tensor)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        
        # 提取文本特征
        text_features = model.encode_text(text_token)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
    
    # 计算每一帧与文本的余弦相似度
    similarities = torch.cosine_similarity(text_features, image_features, dim=1)
    print(similarities)
    
    # 如果只有一帧，确保similarities是一个张量
    if len(frames) == 1:
        similarities = similarities.unsqueeze(0)
    
    # 计算平均相似度作为最终的文本遵循度分数
    avg_similarity = similarities.mean().item()
    
    return avg_similarity

def evaluate_multiple_models(csv_path, output_path, base_dir):
    """
    评估多个模型生成的视频与文本描述的语义一致性
    
    参数:
        csv_path: 输入CSV文件路径
        output_path: 输出结果的CSV文件路径
        base_dir: 所有模型结果文件夹的基础路径
    """
    # 模型名称和对应的结果文件夹
    models = {
        "Ruyi": "ruyi_result",
        "OpenSora": "Open-Sora2.0_result",
        "CogVideoX": "cogVideo2B_result",
        "Wan": "wan1.3B_result"
    }
    
    # 加载CLIP模型
    model, preprocess, device = load_clip_model()
    
    # 读取CSV文件
    video_text_pairs = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                video_filename = row[0] + ".mp4"
                text = row[2][:75]                            # clip一次最多只能处理长度为77的字符串
                video_text_pairs.append((video_filename, text))
    
    # 存储每个视频文件的结果
    results = []
    
    # 遍历每个视频文件
    for video_filename, text in tqdm(video_text_pairs, desc="处理视频文件"):
        result_row = {
            "video_filename": video_filename,
            "text": text
        }
        
        # 对每个模型计算文本遵循度
        for model_name, result_folder in models.items():
            model_result_dir = os.path.join(base_dir, result_folder)
            video_path = os.path.join(model_result_dir, video_filename)
            
            # 检查视频文件是否存在
            if not os.path.exists(video_path):
                print(f"警告: {model_name}模型的视频文件不存在: {video_path}")
                score = 0.0  # 文件不存在时分数设为0
            else:
                score = compute_text_compliance(model, preprocess, device, video_path, text)
            
            # 将该模型的分数添加到结果中
            result_row[f"{model_name}_score"] = score
        
        results.append(result_row)
    
    # 将结果写入新的CSV文件
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ["video_filename", "text", "Ruyi_score", "OpenSora_score", "CogVideoX_score", "Wan_score"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "video_filename": result["video_filename"],
                "text": result["text"],
                "Ruyi_score": result["Ruyi_score"],
                "OpenSora_score": result["OpenSora_score"],
                "CogVideoX_score": result["CogVideoX_score"],
                "Wan_score": result["Wan_score"]
            })
    
    # 计算每个模型的平均分数
    model_avg_scores = {}
    for model_name in models.keys():
        scores = [result[f"{model_name}_score"] for result in results]
        avg_score = sum(scores) / len(scores) if scores else 0
        model_avg_scores[model_name] = avg_score
        print(f"{model_name}模型的平均文本遵循度分数: {avg_score:.4f}")
    
    return results, model_avg_scores

if __name__ == "__main__":
    input_csv = "../True_video/video_caption.csv"
    output_csv = "../video-text_alignment111.csv"
    base_dir = "../"
    
    # 评估多个模型的数据集
    evaluate_multiple_models(input_csv, output_csv, base_dir)