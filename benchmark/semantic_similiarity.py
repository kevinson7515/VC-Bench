# import os
# import torch
# import clip
# from PIL import Image
# import numpy as np
# import cv2
# from tqdm import tqdm
# from sklearn.metrics.pairwise import cosine_similarity
# import matplotlib.pyplot as plt



# class VideoCLIPEvaluator:
#     def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
#         self.device = device
#         self.model, self.preprocess = clip.load("RN50x16", device=device)
#         self.model.eval()
        
#     def extract_video_frames(self, video_path, num_frames=16):
#         """均匀采样视频帧"""
#         cap = cv2.VideoCapture(video_path)
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         frame_indices = np.linspace(0, total_frames-1, num=num_frames, dtype=int)
        
#         frames = []
#         for idx in frame_indices:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#             ret, frame = cap.read()
#             if ret:
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frames.append(Image.fromarray(frame))
#         cap.release()
#         return frames
    
#     def get_clip_features(self, frames, text_prompt=None):
#         """提取CLIP特征"""
#         frame_features = []
#         processed_frames = [self.preprocess(frame).unsqueeze(0).to(self.device) for frame in frames]
        
#         with torch.no_grad():
#             # 提取图像特征
#             for frame in processed_frames:
#                 image_features = self.model.encode_image(frame)
#                 frame_features.append(image_features.cpu().numpy())
            
#             # 提取文本特征
#             if text_prompt:
#                 text_input = clip.tokenize([text_prompt]).to(self.device)
#                 text_features = self.model.encode_text(text_input).cpu().numpy()
#                 return np.array(frame_features), text_features
#             return np.array(frame_features)
    
#     def text_video_alignment(self, video_path, text_prompt):
#         """文本-视频对齐度评估"""
#         frames = self.extract_video_frames(video_path)
#         frame_features, text_features = self.get_clip_features(frames, text_prompt)
        
#         # 计算每帧与文本的相似度
#         similarities = cosine_similarity(
#             text_features.reshape(1, -1),
#             frame_features.reshape(len(frames), -1)
#         )[0]
        
#         return {
#             'mean_similarity': float(np.mean(similarities)),
#             'min_similarity': float(np.min(similarities)),
#             'max_similarity': float(np.max(similarities)),
#             'similarity_curve': similarities.tolist()
#         }
    
#     def video_consistency(self, video_path):
#         """视频内容一致性评估"""
#         frames = self.extract_video_frames(video_path)
#         frame_features = self.get_clip_features(frames)
        
#         # 计算帧间相似度矩阵
#         sim_matrix = cosine_similarity(
#             frame_features.reshape(len(frames), -1))
        
#         # 提取非对角线元素（帧间相似度）
#         mask = ~np.eye(len(frames), dtype=bool)
#         inter_frame_sims = sim_matrix[mask]
        
#         return {
#             'mean_consistency': float(np.mean(inter_frame_sims)),
#             'consistency_matrix': sim_matrix.tolist()
#         }
    
#     def video_diversity(self, video_path):
#         """视频内容多样性评估"""
#         frames = self.extract_video_frames(video_path)
#         frame_features = self.get_clip_features(frames)
        
#         # 计算特征方差
#         feature_var = np.var(frame_features, axis=0).mean()
        
#         # 计算平均帧间距离
#         distances = []
#         for i in range(len(frame_features)):
#             for j in range(i+1, len(frame_features)):
#                 dist = 1 - cosine_similarity(
#                     frame_features[i].reshape(1, -1),
#                     frame_features[j].reshape(1, -1)
#                 )[0][0]
#                 distances.append(dist)
                
#         return {
#             'feature_variance': float(feature_var),
#             'mean_pairwise_distance': float(np.mean(distances))
#         }
    
#     def visualize_results(self, results):
#         """可视化评估结果"""
#         plt.figure(figsize=(15, 5))
        
#         # 文本-视频对齐度曲线
#         if 'similarity_curve' in results['alignment']:
#             plt.subplot(1, 3, 1)
#             plt.plot(results['alignment']['similarity_curve'], 'b-o')
#             plt.title('Text-Video Alignment')
#             plt.xlabel('Frame Index')
#             plt.ylabel('CLIP Similarity')
#             plt.ylim(0, 1)
#             plt.grid()
        
#         # 一致性矩阵
#         if 'consistency_matrix' in results['consistency']:
#             plt.subplot(1, 3, 2)
#             plt.imshow(results['consistency']['consistency_matrix'], cmap='hot')
#             plt.title('Frame Consistency Matrix')
#             plt.colorbar()
        
#         # 多样性指标
#         if 'feature_variance' in results['diversity']:
#             plt.subplot(1, 3, 3)
#             metrics = ['Feature Variance', 'Pairwise Distance']
#             values = [
#                 results['diversity']['feature_variance'],
#                 results['diversity']['mean_pairwise_distance']
#             ]
#             plt.bar(metrics, values)
#             plt.title('Diversity Metrics')
#             plt.ylabel('Value')
        
#         plt.tight_layout()
#         plt.show()

#     def evaluate(self, video_path, text_prompt):
#         """综合评估"""
#         results = {
#             'alignment': self.text_video_alignment(video_path, text_prompt),
#             'consistency': self.video_consistency(video_path),
#             'diversity': self.video_diversity(video_path)
#         }
        
#         # 计算综合评分（加权平均）
#         weights = {'alignment': 0.5, 'consistency': 0.3, 'diversity': 0.2}
#         composite_score = (
#             weights['alignment'] * results['alignment']['mean_similarity'] +
#             weights['consistency'] * results['consistency']['mean_consistency'] +
#             weights['diversity'] * results['diversity']['mean_pairwise_distance']
#         )
#         results['composite_score'] = float(composite_score)
        
#         self.visualize_results(results)
#         return results

# # 使用示例
# if __name__ == "__main__":
#     evaluator = VideoCLIPEvaluator()
#     video_path = "/home/export/base/ycsc_chenkh/hitici_08/online1/Files/VC_benchmark/examples/13155381_1024_576_24fps.mp4"  # 替换为你的视频路径
#     text_prompt = "The video shows a person snowboarding down a snowy slope surrounded by trees. The individual is wearing winter gear, including a black jacket, beige pants, and goggles. They are skillfully maneuvering their snowboard through the fresh snow, carving turns and maintaining balance. The scene is set in a forested area with tall trees covered in snow, creating a picturesque winter landscape."  
#      # 替换为生成视频时的文本提示
    
#     evaluation_results = evaluator.evaluate(video_path, text_prompt)
#     print("\nEvaluation Results:")
#     print(f"Composite Score: {evaluation_results['composite_score']:.3f}/1.0")
#     print(f"Text-Video Alignment: {evaluation_results['alignment']['mean_similarity']:.3f}")
#     print(f"Frame Consistency: {evaluation_results['consistency']['mean_consistency']:.3f}")
#     print(f"Content Diversity: {evaluation_results['diversity']['mean_pairwise_distance']:.3f}")  















import torch.nn as nn
from tkinter import Image
import numpy as np
import open_clip
from regex import F
import torch
import sys
sys.path.append('/home/export/base/ycsc_chenkh/hitici_08/online1/Files/VC_benchmark')
from benchmark.utils import load_video, load_dimension_info, clip_transform
from benchmark.distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)

class EnhancedCLIPEvaluator:
    def __init__(self, model_name='ViT-bigG-14', pretrained='laion2b_s39b_b160k'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载更强大的CLIP模型
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            precision="fp16" if 'cuda' in self.device else "fp32",
            device=self.device
        )
        
        # 初始化tokenizer
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.image_transform = clip_transform(224)

    def get_features(self, images, text_prompt=None):
        """提取特征（支持批量处理）"""
        with torch.no_grad(), torch.cuda.amp.autocast():
            # 图像特征
            total_frames = images.shape[0]
            num_frames = 48
            img_indices = np.linspace(0, total_frames-1, num=num_frames, dtype=int)
            images = images[img_indices]
            images = self.image_transform(images)
            images = images.to(self.device)
            image_features = self.model.encode_image(images)
            image_features =nn.functional.normalize(image_features, dim=-1, p=2)
            # image_features /= image_features.norm(dim=-1, keepdim=True)
            
            if text_prompt:
                # 文本特征
                text = self.tokenizer(text_prompt).to(self.device)
                text_features = self.model.encode_text(text)
                text_features =nn.functional.normalize(text_features, dim=-1, p=2)

                #text_features /= text_features.norm(dim=-1, keepdim=True)
                return image_features, text_features
            return image_features

    def evaluate_alignment(self, video_path, text_prompt):
        """增强版对齐度评估"""
        # 预处理帧（批量处理提升效率）
        images = load_video(video_path).to(self.device)
        
        # 提取特征
        image_features, text_features = self.get_features(images, text_prompt)
        #image_features = torch.mean(image_features, dim=0)

        print(image_features.shape, text_features.shape)
        
        # 计算相似度（矩阵运算优化）
        #similarity = ((image_features @ text_features.T) / (torch.sqrt(torch.inner(image_features, image_features)) * torch.sqrt(torch.inner(text_features, text_features)))).squeeze().cpu().numpy()
        similarity = nn.functional.cosine_similarity(image_features, text_features).squeeze().cpu().numpy()
        return {
            'mean_sim': similarity.max(),
            'dynamic_range': similarity.ptp(),  # 峰值间差异
            'frame_sims': similarity.tolist()
        }

# 使用示例
if __name__ == "__main__":
    # 初始化增强评估器（使用当前开源最强模型）
    evaluator = EnhancedCLIPEvaluator(
        model_name="ViT-bigG-14",
        pretrained="/home/export/base/ycsc_chenkh/hitici_08/online1/.cache/clip/open_clip_pytorch_model.bin"
    )
    
    # 模拟输入
    json_dir = '/home/export/base/ycsc_chenkh/hitici_08/online1/Files/VC_benchmark/examples/ori_12688805_576_1024_24fps.mp4'
    # text_prompt = "The video shows a person riding a motorcycle on a road surrounded by trees. The camera is mounted on the handlebars, providing a first-person perspective of the ride. The speedometer and other gauges are visible in the center of the frame, indicating the speed and other information about the motorcycle's performance. The road appears to be winding through a forested area, with greenery on both sides. The rider's hands can be seen gripping the handlebars as they navigate the twists and turns of the road. Overall, the video captures the thrill and excitement of riding a motorcycle through a scenic natural environment."
    #text_prompt = "The video captures a serene autumn scene in a park. The ground is covered with fallen yellow leaves, creating a vibrant carpet that stretches across the pathway and grassy areas. Tall trees with golden foliage line the path, their branches reaching out towards each other, forming a natural canopy overhead. The sky above is overcast, casting a soft, diffused light over the entire scene. In the background, there are more trees, some of which have lost most of their leaves, adding to the sense of the changing season. The overall atmosphere is peaceful and inviting, evoking a sense of calm and tranquility."
    text_prompt = "fallen yellow leaves, the pathway and grassy areas,  golden foliage"
    # 评估
    results = evaluator.evaluate_alignment(json_dir, text_prompt)
    print(f"文本-视频对齐度: {results['mean_sim']:.3f} (范围: {results['dynamic_range']:.3f})")
