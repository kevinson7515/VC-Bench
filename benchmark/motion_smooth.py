import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import warnings
warnings.filterwarnings("ignore")



def caldist(a, b, c, d):
    return abs(a-c) + abs(b-d)



def feature_tracking(video_path, max_corners=1000, min_track_length=20):
    """
    基于KLT特征点跟踪的视频运动平滑度分析
    :param video_path: 输入视频路径
    :param max_corners: 最大跟踪特征点数
    :param min_track_length: 有效轨迹的最小长度（帧数）
    :return: smoothness_score (0~1), trajectories (所有轨迹数据)
    """
    # 初始化视频捕获
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")

    # KLT跟踪器参数
    lk_params = dict(
        winSize=(10, 10),  # 搜索窗口大小
        maxLevel=3,        # 金字塔层数
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    # 读取第一帧并初始化特征点
    ret, old_frame = cap.read()
    if not ret:
        raise ValueError("视频读取失败")
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    # 使用Shi-Tomasi角点检测初始特征点
    p0 = cv2.goodFeaturesToTrack(
        old_gray,
        maxCorners=max_corners,
        qualityLevel=0.0001,
        minDistance=5,
        blockSize=3
    )
    if p0 is None:
        raise ValueError("未检测到有效特征点")

    # 初始化轨迹存储
    trajectories = [[] for _ in range(len(p0))]
    frame_count = 0
    colors = np.random.randint(0, 255, (max_corners, 3))  # 为每个轨迹分配随机颜色

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 计算光流（跟踪特征点）
        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )
        if p1 is None:
        #    p0 = cv2.goodFeaturesToTrack(
        #    frame_gray,
        #    maxCorners=max_corners,
        #    qualityLevel=0.0001,
        #    minDistance=10,
        #    blockSize=3
        #  )
        #    old_frame = frame_gray
           #raise ValueError("未检测到有效特征点")
           continue
           
        
        # 只保留跟踪成功的点
        good_new = p1[st == 1]
        good_old = p0[st == 1]



        inits = 0
        for i, (new, old) in enumerate(zip(good_new,good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            distance = caldist(a, b, c, d)
            if distance < 3:
                good_new[inits] = good_new[i]
                good_old[inits] = good_old[i]
                inits += 1
        good_new = good_new[:inits]
        good_old = good_old[:inits]

        
        # 更新轨迹
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            x_new, y_new = new.ravel()
            x_old, y_old = old.ravel()
            trajectories[i].append((x_new, y_new))
        
        # 更新前一帧和特征点
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        frame_count += 1

    cap.release()
    #cv2.destroyAllWindows()

    # 过滤过短的轨迹
    #valid_trajectories = [t for t in trajectories if len(t) >= min_track_length]
    # if not valid_trajectories:
    #     raise ValueError("没有足够长的有效轨迹")
    return trajectories

def calculate_smoothness(trajectories):
    """
    计算轨迹平滑度得分（0~1，越高越平滑）
    :param trajectories: 轨迹列表，每个轨迹是[(x1,y1), (x2,y2), ...]
    :return: 平均平滑度得分
    """
    smoothness_scores = []
    
    for traj in trajectories:

        if len(traj)<3: 
            continue
        # 将轨迹转换为numpy数组
        traj_array = np.array(traj)
        x, y = traj_array[:, 0], traj_array[:, 1]
        
        # 方法1：速度变化率（加速度）的倒数
        dx = np.diff(x)
        dy = np.diff(y)
        d2x = np.diff(dx)
        d2y = np.diff(dy)
        #velocity = np.sqrt(dx**2 + dy**2)
        #acceleration = np.diff(velocity)
        acc = np.sqrt(d2x**2 + d2y**2)
        score1 = np.mean(1 / (1 + acc))
        # print('111',acc)
        
        # 方法2：轨迹曲率（二阶导数）的倒数
        time = np.arange(len(x))
        coeffs_x = np.polyfit(time, x, 3)  # 二次多项式拟合
        coeffs_y = np.polyfit(time, y, 3)
        fitted_x = np.polyval(coeffs_x, time)
        fitted_y = np.polyval(coeffs_y, time)
        residual = np.mean((x - fitted_x)**2 + (y - fitted_y)**2)
        score2 = 1 / (1 + residual)
        # print('222',residual)
        # 综合评分
        smoothness_scores.append(0.9 * score1 + 0.1 * score2)
    
    return np.mean(smoothness_scores)

# def plot_trajectories(trajectories, sample_index=0):
#     """ 绘制示例轨迹及其速度/加速度曲线 """
#     if sample_index >= len(trajectories):
#         sample_index = 0
    
#     traj = np.array(trajectories[sample_index])
#     t = np.arange(len(traj))
#     x, y = traj[:, 0], traj[:, 1]
    
#     # 计算速度和加速度
#     dx = np.diff(x)
#     dy = np.diff(y)
#     velocity = np.sqrt(dx**2 + dy**2)
#     acceleration = np.diff(velocity)
    
#     # 绘图
#     plt.figure(figsize=(12, 8))
    
#     plt.subplot(2, 2, 1)
#     plt.plot(x, y, 'b-', label='Trajectory')
#     plt.title('Feature Point Path')
#     plt.xlabel('X Position')
#     plt.ylabel('Y Position')
#     plt.grid(True)
    
#     plt.subplot(2, 2, 2)
#     plt.plot(t[:-1], velocity, 'g-', label='Velocity')
#     plt.title('Velocity Over Time')
#     plt.xlabel('Frame Index')
#     plt.ylabel('Velocity (pixels/frame)')
#     plt.grid(True)
    
#     plt.subplot(2, 2, 3)
#     plt.plot(t[:-2], acceleration, 'r-', label='Acceleration')
#     plt.title('Acceleration Over Time')
#     plt.xlabel('Frame Index')
#     plt.ylabel('Acceleration (pixels/frame²)')
#     plt.grid(True)
    
#     plt.subplot(2, 2, 4)
#     freqs, psd = signal.welch(velocity, fs=30, nperseg=min(256, len(velocity)))
#     plt.semilogy(freqs, psd, 'm-')
#     plt.title('Power Spectral Density')
#     plt.xlabel('Frequency [Hz]')
#     plt.ylabel('PSD')
#     plt.grid(True)
    
#     plt.tight_layout()
#     # plt.show()

# 使用示例
if __name__ == "__main__":
    video_path = "/home/export/base/ycsc_chenkh/hitici_08/online1/Files/VC_benchmark/examples/00000027_5s.mp4"  # 替换为你的视频路径
    
    try:
        trajectories = feature_tracking(video_path)
        video_score = calculate_smoothness(trajectories)
        print(f"视频运动平滑度得分: {video_score:.3f} (0~1, 越高越平滑)")
        
        
    except Exception as e:
        print(f"错误: {str(e)}")