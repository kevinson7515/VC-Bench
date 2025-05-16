import cv2
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity

# 初始化VGG模型（去掉全连接层，使用卷积层输出）
def load_vgg_feature_extractor(device='cuda'):
    vgg = models.vgg16(pretrained=True).features  # 只使用卷积部分
    vgg = vgg.to(device).eval()  # 设为评估模式
    for param in vgg.parameters():
        param.requires_grad = False
    return vgg

# 图像预处理
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # VGG输入尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 提取单帧特征
def extract_feature(frame, model, device):
    frame_tensor = preprocess(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(frame_tensor)
    return features.cpu().numpy().flatten()  # 展平为向量

# 计算背景一致性得分
def evaluate_background_consistency(video_path, stride=5):
    # 初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg = load_vgg_feature_extractor(device)
    cap = cv2.VideoCapture(video_path)
    features = []
    frame_indices = []
    
    # 逐帧处理（按stride跳帧加速）
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            feature = extract_feature(frame_rgb, vgg, device)
            features.append(feature)
            frame_indices.append(idx)
        idx += 1
    
    # 计算帧间相似度矩阵
    features = np.array(features)
    sim_matrix = cosine_similarity(features)
    
    # 分析相似度变化
    mean_similarities = np.mean(sim_matrix, axis=1)
    consistency_score = np.mean(mean_similarities)  # 整体一致性得分
    std_similarities = np.std(mean_similarities)    # 波动程度
    
    # 检测异常帧（相似度突降）
    threshold = np.median(mean_similarities) - 2 * std_similarities
    anomalies = np.where(mean_similarities < threshold)[0]
    
    return {
        'consistency_score': consistency_score,  # 越高背景越稳定
        'std_deviation': std_similarities,      # 波动越大越不稳定
        'anomaly_frames': frame_indices[anomalies] if len(anomalies) > 0 else [],
        'similarity_matrix': sim_matrix         # 用于可视化
    }

# 使用示例
if __name__ == "__main__":
    video_path = "/home/export/base/ycsc_chenkh/hitici_08/online1/Files/VC_benchmark/examples/ori_12314475_1024_576_24fps.mp4"
    results = evaluate_background_consistency(video_path, stride=10)
    
    print(f"背景一致性得分: {results['consistency_score']:.3f}")
    print(f"波动标准差: {results['std_deviation']:.3f}")
    if results['anomaly_frames']:
        print(f"检测到异常帧: {results['anomaly_frames']}")
    else:
        print("背景一致性良好，无异常帧")

        