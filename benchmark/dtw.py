import numpy as np
import torch
import clip
from torchvision import transforms
import cv2
from tqdm import tqdm

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Function to extract features from a video frame
def extract_features(frame):
    # Convert BGR to RGB
    #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Preprocess the image for CLIP
    image = preprocess(transforms.ToPILImage()(frame)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features.squeeze(0)

# Function to read video and extract features
def video_to_features(video):
    features = []
    for frame in video:
        features.append(extract_features(frame))
    
    return features




# def load_video(video_path):
#     video = []
#     cap = cv2.VideoCapture(video_path)
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if ret:
#             video.append(frame)
#         else:
#             break
#     cap.release()
        
#     return video





# Cosine similarity distance function
def cosine_distance(a, b):
    # We convert cosine similarity to a "distance" by subtracting from 1
    # Since we want smaller values to indicate more similarity
    return 1 - torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))

def dtw(x, y, dist=cosine_distance):
    """
    Computes Dynamic Time Warping (DTW) of two video sequences using CLIP features.
    
    Parameters:
    x : list of torch.Tensor
        CLIP features of frames from first video
    y : list of torch.Tensor
        CLIP features of frames from second video
    dist : function
        Distance function. Default is cosine distance.
        
    Returns:
    DTW path between x and y.
    """
    # Create cost matrix
    cost = np.zeros((len(x), len(y)))

    # Initialize the first cell
    cost[0, 0] = dist(x[0], y[0]).item()

    # Initialize the first column
    for i in range(1, len(x)):
        cost[i, 0] = cost[i-1, 0] + dist(x[i], y[0]).item()

    # Initialize the first row
    for j in range(1, len(y)):
        cost[0, j] = cost[0, j-1] + dist(x[0], y[j]).item()

    # Populate the rest of the cost matrix
    for i in range(1, len(x)):
        for j in range(1, len(y)):
            choices = cost[i-1, j], cost[i, j-1], cost[i-1, j-1]
            cost[i, j] = dist(x[i], y[j]).item() + min(choices)

    # print("Cost matrix:")
    # print(cost)
    
    # Backtrack to find the optimal path
    path = []
    i, j = len(x)-1, len(y)-1
    path.append([i, j])
    
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            min_val = min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
            if cost[i-1, j] == min_val:
                i -= 1
            elif cost[i, j-1] == min_val:
                j -= 1
            else:
                i -= 1
                j -= 1
        path.append([i, j])
    
    # Reverse the path to get it from start to end
    path = path[::-1]
    
    return path








# Example usage
if __name__ == "__main__":
    video_path1 = "/home/export/base/ycsc_chenkh/hitici_08/online1/Files/VC_benchmark/examples/ori_12314475_1024_576_24fps.mp4"
    video_path2 = "/home/export/base/ycsc_chenkh/hitici_08/online1/Files/VC_benchmark/examples/gen_12314475_1024_576_24fps.mp4"
    
    print("Extracting features from video 1...")
    x_features = video_to_features(video_path1)
    print("Extracting features from video 2...")
    y_features = video_to_features(video_path2)
    
    print("Computing DTW...")
    path = dtw(x_features, y_features)
    print("DTW path:")
    print(path)