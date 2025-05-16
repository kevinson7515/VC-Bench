# # 值越小，视频越相似
# import tensorflow as tf
# import numpy as np
# from scipy.linalg import sqrtm
# import cv2
# import os
# import csv
# import math

# class FVDCalculator:
#     def __init__(self, model_path="i3d-kinetics-tensorflow1-400-v1"):
#         """Initialize with local I3D model"""
#         # Load local I3D model
#         if not os.path.exists(model_path):
#             raise ValueError(f"I3D model directory not found at: {model_path}")
        
#         # Load the model from local directory with correct tags
#         try:
#             # First try loading with empty tags (as per error message)
#             self.i3d_model = tf.saved_model.load(model_path, tags=[])
#             # Try to get the default signature (may have different name)
#             if not hasattr(self.i3d_model, 'signatures'):
#                 raise ValueError("Loaded model doesn't have signatures attribute")
            
#             # Try common signature names
#             for sig_name in ['default', 'inference', 'predict', 'serving_default']:
#                 if sig_name in self.i3d_model.signatures:
#                     self.i3d_model = self.i3d_model.signatures[sig_name]
#                     break
#             else:
#                 # If no known signature found, use the first available one
#                 if len(self.i3d_model.signatures) > 0:
#                     self.i3d_model = self.i3d_model.signatures[next(iter(self.i3d_model.signatures))]
#                 else:
#                     raise ValueError("No signatures found in the model")
                    
#         except Exception as e:
#             try:
#                 # If that fails, try with 'train' tag (as indicated in the error)
#                 self.i3d_model = tf.saved_model.load(model_path, tags=['train'])
#                 if not hasattr(self.i3d_model, 'signatures'):
#                     raise ValueError("Loaded model doesn't have signatures attribute")
                
#                 # Try common signature names
#                 for sig_name in ['default', 'inference', 'predict', 'serving_default']:
#                     if sig_name in self.i3d_model.signatures:
#                         self.i3d_model = self.i3d_model.signatures[sig_name]
#                         break
#                 else:
#                     if len(self.i3d_model.signatures) > 0:
#                         self.i3d_model = self.i3d_model.signatures[next(iter(self.i3d_model.signatures))]
#                     else:
#                         raise ValueError("No signatures found in the model")
                        
#             except Exception as e:
#                 raise ValueError(f"Failed to load model from {model_path}. Error: {str(e)}")
    
#     def load_video(self, video_path):
#         """Load video and extract all frames using OpenCV"""
#         cap = cv2.VideoCapture(video_path)
#         frames = []
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame = cv2.resize(frame, (224, 224))
#             frames.append(frame)
        
#         cap.release()
#         return np.array(frames)

#     def preprocess(self, frames):
#         """Normalize frames for I3D input"""
#         frames = frames.astype(np.float32) / 255.0
#         return tf.convert_to_tensor(frames[np.newaxis, ...])

#     def get_features(self, video_path):
#         """Extract I3D features for all 16-frame segments with padding"""
#         frames = self.load_video(video_path)
#         features = []
        
#         # Process video in 16-frame segments with padding if needed
#         for i in range(0, len(frames), 16):
#             segment = frames[i:i+16]
            
#             # Pad with last frame if segment is shorter than 16 frames
#             if len(segment) < 16:
#                 padding = np.tile(frames[-1:], (16 - len(segment), 1, 1, 1))
#                 segment = np.concatenate([segment, padding], axis=0)
                
#             preprocessed = self.preprocess(segment)
#             # Use the local model to extract features
#             # Try both 'default' and other possible output names
#             try:
#                 output = self.i3d_model(preprocessed)
#                 if 'default' in output:
#                     feat = output['default'].numpy()
#                 elif 'predictions' in output:
#                     feat = output['predictions'].numpy()
#                 elif 'features' in output:
#                     feat = output['features'].numpy()
#                 else:
#                     # Use the first output if name is unknown
#                     feat = list(output.values())[0].numpy()
                    
#                 features.append(feat)
#             except Exception as e:
#                 raise ValueError(f"Failed to extract features: {str(e)}")
        
#         return np.concatenate(features, axis=0) if features else np.zeros((1, 1024))

#     def calculate_fvd(self, real_path, generated_path):
#         """Calculate FVD between two videos by averaging over all 16-frame segments"""
#         # Get features (shape will be [num_segments, feature_dim])
#         real_features = self.get_features(real_path)
#         gen_features = self.get_features(generated_path)
        
#         # If one video has more segments than the other, truncate to the smaller number
#         min_segments = min(len(real_features), len(gen_features))
#         if min_segments == 0:
#             return float('inf')
            
#         real_features = real_features[:min_segments]
#         gen_features = gen_features[:min_segments]
        
#         # Calculate statistics
#         mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
#         mu_gen, sigma_gen = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
        
#         # Fréchet distance calculation
#         diff = mu_real - mu_gen
#         covmean = sqrtm(sigma_real.dot(sigma_gen))
        
#         if np.iscomplexobj(covmean):
#             covmean = covmean.real
        
#         return np.sum(diff**2) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    
#     def calculate_fvd_gauss(self, video_path):
#         """Calculate FVD between real video features and a Gaussian distribution"""
#         real_features = self.get_features(video_path)
#         if len(real_features) == 0:
#             return float('inf')
        
#         # Calculate statistics of real features
#         mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        
#         # Create Gaussian distribution with same dimension but zero mean and identity covariance
#         dim = mu_real.shape[0]
#         mu_gauss = np.zeros(dim)
#         sigma_gauss = np.eye(dim)
        
#         # Calculate FVD between real features and Gaussian distribution
#         fvd_score = self._compute_frechet_distance(mu_real, sigma_real, mu_gauss, sigma_gauss)
#         return fvd_score
    
#     def _compute_frechet_distance(self, mu1, sigma1, mu2, sigma2):
#         """Compute the Fréchet distance between two multivariate Gaussians"""
#         diff = mu1 - mu2
#         covmean = sqrtm(sigma1.dot(sigma2))
        
#         if np.iscomplexobj(covmean):
#             covmean = covmean.real
        
#         return np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean)


# if __name__ == "__main__":
#     fvd_calculator = FVDCalculator(model_path="i3d-kinetics-tensorflow1-400-v1")
#     real_video = "./origin_video.mp4"
#     generated_video = "./generated_video.mp4"
#     score_fvd = fvd_calculator.calculate_fvd(real_video, generated_video)
#     score_real_gauss = fvd_calculator.calculate_fvd_gauss(real_video)
#     score_generated_gauss = fvd_calculator.calculate_fvd_gauss(generated_video)
#     semantic_similarity = 1 - score_fvd / math.sqrt(score_real_gauss * score_generated_gauss)
#     print(semantic_similarity)

import os
import re
import csv
import math
import numpy as np
import cv2
from scipy.linalg import sqrtm
import tensorflow as tf
from tqdm import tqdm

class FVDCalculator:
    def __init__(self, model_path="i3d-kinetics-tensorflow1-400-v1"):
        """Initialize with local I3D model"""
        # Load local I3D model
        if not os.path.exists(model_path):
            raise ValueError(f"I3D model directory not found at: {model_path}")
        
        # Load the model from local directory with correct tags
        try:
            # First try loading with empty tags (as per error message)
            self.i3d_model = tf.saved_model.load(model_path, tags=[])
            # Try to get the default signature (may have different name)
            if not hasattr(self.i3d_model, 'signatures'):
                raise ValueError("Loaded model doesn't have signatures attribute")
            
            # Try common signature names
            for sig_name in ['default', 'inference', 'predict', 'serving_default']:
                if sig_name in self.i3d_model.signatures:
                    self.i3d_model = self.i3d_model.signatures[sig_name]
                    break
            else:
                # If no known signature found, use the first available one
                if len(self.i3d_model.signatures) > 0:
                    self.i3d_model = self.i3d_model.signatures[next(iter(self.i3d_model.signatures))]
                else:
                    raise ValueError("No signatures found in the model")
                    
        except Exception as e:
            try:
                # If that fails, try with 'train' tag (as indicated in the error)
                self.i3d_model = tf.saved_model.load(model_path, tags=['train'])
                if not hasattr(self.i3d_model, 'signatures'):
                    raise ValueError("Loaded model doesn't have signatures attribute")
                
                # Try common signature names
                for sig_name in ['default', 'inference', 'predict', 'serving_default']:
                    if sig_name in self.i3d_model.signatures:
                        self.i3d_model = self.i3d_model.signatures[sig_name]
                        break
                else:
                    if len(self.i3d_model.signatures) > 0:
                        self.i3d_model = self.i3d_model.signatures[next(iter(self.i3d_model.signatures))]
                    else:
                        raise ValueError("No signatures found in the model")
                        
            except Exception as e:
                raise ValueError(f"Failed to load model from {model_path}. Error: {str(e)}")
    
    def load_video(self, video_path):
        """Load video and extract all frames using OpenCV"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        
        cap.release()
        return np.array(frames)

    def preprocess(self, frames):
        """Normalize frames for I3D input"""
        frames = frames.astype(np.float32) / 255.0
        return tf.convert_to_tensor(frames[np.newaxis, ...])

    def get_features(self, video_path):
        """Extract I3D features for all 16-frame segments with padding"""
        frames = self.load_video(video_path)
        features = []
        
        # Process video in 16-frame segments with padding if needed
        for i in range(0, len(frames), 16):
            segment = frames[i:i+16]
            
            # Pad with last frame if segment is shorter than 16 frames
            if len(segment) < 16:
                padding = np.tile(frames[-1:], (16 - len(segment), 1, 1, 1))
                segment = np.concatenate([segment, padding], axis=0)
                
            preprocessed = self.preprocess(segment)
            # Use the local model to extract features
            # Try both 'default' and other possible output names
            try:
                output = self.i3d_model(preprocessed)
                if 'default' in output:
                    feat = output['default'].numpy()
                elif 'predictions' in output:
                    feat = output['predictions'].numpy()
                elif 'features' in output:
                    feat = output['features'].numpy()
                else:
                    # Use the first output if name is unknown
                    feat = list(output.values())[0].numpy()
                    
                features.append(feat)
            except Exception as e:
                raise ValueError(f"Failed to extract features: {str(e)}")
        
        return np.concatenate(features, axis=0) if features else np.zeros((1, 1024))

    def _normalize_features(self, features):
        """Normalize features to [0, 1] range"""
        if len(features) == 0:
            return features
            
        # Min-max normalization
        min_val = np.min(features)
        max_val = np.max(features)
        
        # Avoid division by zero if all values are same
        if max_val - min_val > 0:
            normalized = (features - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(features)
            
        return normalized

    def calculate_fvd(self, real_path, generated_path):
        """Calculate FVD between two videos by averaging over all 16-frame segments"""
        # Get features (shape will be [num_segments, feature_dim])
        real_features = self.get_features(real_path)
        gen_features = self.get_features(generated_path)
        
        # Normalize features to [0, 1] range
        real_features = self._normalize_features(real_features)
        gen_features = self._normalize_features(gen_features)
        
        # If one video has more segments than the other, truncate to the smaller number
        min_segments = min(len(real_features), len(gen_features))
        if min_segments == 0:
            return float('inf')
            
        real_features = real_features[:min_segments]
        gen_features = gen_features[:min_segments]
        
        # Calculate statistics
        mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu_gen, sigma_gen = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
        
        # Fréchet distance calculation
        diff = mu_real - mu_gen
        covmean = sqrtm(sigma_real.dot(sigma_gen))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        return np.sum(diff**2) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    
    def calculate_fvd_gauss(self, video_path):
        """Calculate FVD between real video features and a Gaussian distribution"""
        real_features = self.get_features(video_path)
        if len(real_features) == 0:
            return float('inf')
        
        # Normalize features to [0, 1] range
        real_features = self._normalize_features(real_features)
        
        # Calculate statistics of real features
        mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        
        # Create Gaussian distribution with same dimension but zero mean and identity covariance
        dim = mu_real.shape[0]
        mu_gauss = np.zeros(dim)
        sigma_gauss = np.eye(dim)
        
        # Calculate FVD between real features and Gaussian distribution
        fvd_score = self._compute_frechet_distance(mu_real, sigma_real, mu_gauss, sigma_gauss)
        return fvd_score
    
    def _compute_frechet_distance(self, mu1, sigma1, mu2, sigma2):
        """Compute the Fréchet distance between two multivariate Gaussians"""
        diff = mu1 - mu2
        covmean = sqrtm(sigma1.dot(sigma2))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        return np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean)

    def calculate_semantic_similarity(self, real_path, generated_path):
        """Calculate semantic similarity between two videos"""
        score_fvd = self.calculate_fvd(real_path, generated_path)
        score_real_gauss = self.calculate_fvd_gauss(real_path)
        score_generated_gauss = self.calculate_fvd_gauss(generated_path)
        
        # Ensure we don't divide by zero
        if score_real_gauss == 0 or score_generated_gauss == 0:
            return 0.0
            
        semantic_similarity = 1 - score_fvd / math.sqrt(score_real_gauss * score_generated_gauss)
        return semantic_similarity

def get_video_frame_count(video_path):
    """Get the number of frames in a video"""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def extract_video_segment(input_path, output_path, start_frame=0, num_frames=None, from_end=False):
    """
    Extract a segment from a video
    
    Args:
        input_path: Path to input video
        output_path: Path to save extracted segment
        start_frame: Frame to start from (if not from_end)
        num_frames: Number of frames to extract
        from_end: If True, extract from the end of the video
    """
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # If extracting from end, calculate start frame
    if from_end and num_frames is not None:
        start_frame = max(0, frame_count - num_frames)
    
    # Ensure we don't go beyond video length
    if num_frames is None or start_frame + num_frames > frame_count:
        num_frames = frame_count - start_frame
    
    # Setup writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Set starting position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Extract frames
    frames_written = 0
    while frames_written < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frames_written += 1
    
    # Release resources
    cap.release()
    out.release()
    
    return frames_written

def process_videos(true_video_dir, generated_video_dir, output_csv, temp_dir="temp"):
    """
    Process video pairs and calculate semantic similarity
    
    Args:
        true_video_dir: Directory containing true head/tail videos
        generated_video_dir: Directory containing generated videos
        output_csv: Path to save CSV results
        temp_dir: Directory for temporary files
    """
    # Create temp directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)
    
    # Initialize FVD calculator
    fvd_calculator = FVDCalculator(model_path="i3d-kinetics-tensorflow1-400-v1")
    
    # Dictionary to store results
    results = {}
    
    # Get all true videos
    true_videos = [f for f in os.listdir(true_video_dir) if f.endswith(".mp4")]
    
    # Group videos by base name
    video_groups = {}
    for video in true_videos:
        # Extract base name (without _head or _tail)
        if "_head" in video:
            base_name = video.replace("_head.mp4", "")
            video_type = "head"
        elif "_tail" in video:
            base_name = video.replace("_tail.mp4", "")
            video_type = "tail"
        else:
            continue  # Skip if not head or tail
            
        if base_name not in video_groups:
            video_groups[base_name] = {"head": None, "tail": None}
        
        video_groups[base_name][video_type] = os.path.join(true_video_dir, video)
    
    # Process each video group
    print(f"Found {len(video_groups)} video groups to process")
    
    for base_name, videos in tqdm(video_groups.items(), desc="Processing videos"):
        # Find the corresponding generated video
        generated_video = os.path.join(generated_video_dir, f"{base_name}.mp4")
        
        if not os.path.exists(generated_video):
            print(f"Warning: Generated video not found for {base_name}")
            continue
            
        similarities = []
        
        # Process head video if available
        if videos["head"] is not None:
            head_path = videos["head"]
            
            # Get frame count
            head_frames = get_video_frame_count(head_path)
            
            # Extract corresponding frames from generated video
            temp_gen_head = os.path.join(temp_dir, f"{base_name}_gen_head.mp4")
            extract_video_segment(generated_video, temp_gen_head, start_frame=0, num_frames=head_frames)
            
            # Calculate semantic similarity
            head_similarity = fvd_calculator.calculate_semantic_similarity(head_path, temp_gen_head)
            similarities.append(head_similarity)
            
            # Cleanup
            if os.path.exists(temp_gen_head):
                os.remove(temp_gen_head)
                
        # Process tail video if available
        if videos["tail"] is not None:
            tail_path = videos["tail"]
            
            # Get frame count
            tail_frames = get_video_frame_count(tail_path)
            
            # Extract corresponding frames from generated video
            temp_gen_tail = os.path.join(temp_dir, f"{base_name}_gen_tail.mp4")
            extract_video_segment(generated_video, temp_gen_tail, num_frames=tail_frames, from_end=True)
            
            # Calculate semantic similarity
            tail_similarity = fvd_calculator.calculate_semantic_similarity(tail_path, temp_gen_tail)
            similarities.append(tail_similarity)
            
            # Cleanup
            if os.path.exists(temp_gen_tail):
                os.remove(temp_gen_tail)
        
        # Calculate average similarity
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            results[base_name] = avg_similarity
            print(f"Video {base_name}: Semantic Similarity = {avg_similarity:.4f}")
    
    # Write results to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Video', 'Semantic_Similarity'])
        
        for base_name, similarity in results.items():
            writer.writerow([base_name, similarity])
    
    print(f"Results saved to {output_csv}")
    
    # Cleanup temp directory
    if os.path.exists(temp_dir):
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)

if __name__ == "__main__":
    true_dir = "../True_video"
    gen_dir = "../Open-Sora2.0_result"
    output_dir = "../semantic_similarity_opensora.csv"
    temp_dir = "../tmp_dir"

    process_videos(
        true_video_dir=true_dir,
        generated_video_dir=gen_dir,
        output_csv=output_dir,
        temp_dir=temp_dir
    )
