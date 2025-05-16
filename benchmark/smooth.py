import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")
# 读取视频
cap = cv2.VideoCapture("/home/export/base/ycsc_chenkh/hitici_08/online1/Files/VC_benchmark/examples/00000025_3s.mp4")
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# 光流参数
flow_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
smoothness_scores = []

magnitude_list = []
angle_list = []

while True:
    ret, curr_frame = cap.read()
    if not ret:
        break
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # 计算稠密光流
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **flow_params)
    #u, v = flow[..., 0], flow[..., 1]
    u, v = flow[::3, ::3, 0], flow[::3,::3, 1]
    
    # 计算运动矢量幅度和方向
    magnitude = np.sqrt(u**2 + v**2)
    angle = np.arctan2(v, u) * 180 / np.pi


    magnitude_list.append(magnitude)
    angle_list.append(angle)

    
    # 指标1：运动矢量幅度标准差（波动越小越平滑）
# magnitude_np = np.array(magnitude_list)

# #mag_std = np.std(magnitude_np, axis=0)
# bin_num = 10
# mag_hist = np.zeros((magnitude_np.shape[1], magnitude_np.shape[2], bin_num))
# for i in range(magnitude_np.shape[1]):
#     for j in range(magnitude_np.shape[2]):
#         mag_hist[i,j], _ = np.histogram(magnitude_np[:,i,j], bins=bin_num)
# mag_prob = mag_hist / magnitude_np.shape[0]
# mag_entropy = -np.sum(mag_prob * np.log2(mag_prob + 1e-10), axis=2)
# print(np.mean(mag_entropy), np.std(mag_entropy))



# angle_np = np.array(angle_list)

# #angle_std = np.std(angle_np, axis=0)
# bin_num = 10
# angle_hist = np.zeros((angle_np.shape[1], angle_np.shape[2], bin_num))
# for i in range(angle_np.shape[1]):
#     for j in range(angle_np.shape[2]):
#         angle_hist[i,j], _ = np.histogram(angle_np[:,i,j], bins=bin_num, range=(0, 360))
# angle_prob = angle_hist / angle_np.shape[0]
# angle_entropy = -np.sum(angle_prob * np.log2(angle_prob + 1e-10), axis=2)
# print(np.mean(angle_entropy), np.std(angle_entropy))


    # 指标2：运动方向熵（熵越低越平滑）
# hist, _ = np.histogram(angle_list, bins=36, range=(0, 360))


    # du = np.diff(u)
    # dv = np.diff(v)
    # acc = np.sqrt(du**2 + dv**2)
    # score = (1 / (1 + np.mean(acc)))
        
    
# smoothness_scores.append((mag_std, entropy, score))
#     prev_gray = curr_gray
# smoothness_scores.append(mag_std)


np.set_printoptions(
    threshold=np.inf,  # 完全禁用省略
    linewidth=np.inf,  # 不换行（避免自动分行省略）
    precision=3       # 可选：控制小数位数
)


magnitude_np = np.array(magnitude_list)
angle_np = np.array(angle_list)
d_magnitude = np.abs(magnitude_np[1:] - magnitude_np[:-1]) 
d_angle = np.abs(angle_np[1:] - angle_np[:-1])

print([ np.sum(d_magnitude[i]>2) for i in range(d_magnitude.shape[0])])







# 输出结果
# print("帧间运动波动（标准差）:", [x[0] for x in smoothness_scores])
# print("帧间运动波动（标准差）平均：", np.mean([x[0] for x in smoothness_scores]))
# print("运动方向熵:", [x[1] for x in smoothness_scores])
# print("运动方向熵:", np.mean([x[1] for x in smoothness_scores]))
# print('加速度：', np.mean([x[2] for x in smoothness_scores]))

#print("帧间运动波动（标准差）平均：", smoothness_scores[0].shape)