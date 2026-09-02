import cv2
import numpy as np
video_path = r'E:\data\0111_testdata\record\V1\Visual\DJI_0844_W.MP4'

# 读取视频
cap = cv2.VideoCapture(video_path)

# 读取第一帧并检测特征点
ret, frame1 = cap.read()
prev_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# 创建遮罩，避开水印区域
mask = np.ones_like(prev_frame)
mask[int(prev_frame.shape[0]*0.25):int(prev_frame.shape[0]*0.75),
     int(prev_frame.shape[1]*0.25):int(prev_frame.shape[1]*0.75)] = 0

prev_pts = cv2.goodFeaturesToTrack(prev_frame, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)

# 创建一个空列表来保存运动参数
motion_params = []

while True:
    # 读取下一帧
    ret, frame2 = cap.read()
    if not ret:
        break

    next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 如果特征点数量低于阈值，则在当前帧重新检测特征点
    if len(prev_pts) < 50:
        prev_pts = cv2.goodFeaturesToTrack(prev_frame, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)

    # 使用Lucas-Kanade方法来估计特征点的运动
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, next_frame, prev_pts, None)

    # 只保留成功跟踪的特征点
    good_new = next_pts[status == 1]
    good_old = prev_pts[status == 1]

    # 使用RANSAC算法来估计单应性矩阵
    H, _ = cv2.findHomography(good_old, good_new, cv2.RANSAC)

    # 保存运动参数
    motion_params.append(H)

    # 更新前一帧和特征点
    prev_frame = next_frame.copy()
    prev_pts = good_new.reshape(-1, 1, 2)

print(motion_params)