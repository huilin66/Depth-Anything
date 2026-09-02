import cv2
import numpy as np
import matplotlib.pyplot as plt


video_path = r'E:\data\0111_testdata\record\V1\Visual\DJI_0844_W.MP4'
# 创建ORB检测器
orb = cv2.ORB_create()

# 读取视频
cap = cv2.VideoCapture(video_path)

color = np.random.randint(0, 255, (200, 3))
# 读取第一帧
ret, frame = cap.read()
prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# 创建遮罩，避开水印区域
mask = np.ones_like(prev_frame)
mask[0:1000, 0:1000] = 0  # 假设水印位于这个区域
# 在第一帧中检测特征点
prev_pts = cv2.goodFeaturesToTrack(prev_frame, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3, mask=mask)

# 设置Lucas-Kanade方法的参数
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 保存每一帧的特征点运动
motion_history = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 使用Lucas-Kanade方法来估计特征点的运动
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_frame, next_frame, prev_pts, None, **lk_params)

    # 只保留成功跟踪的特征点
    good_new = next_pts[status == 1]
    good_old = prev_pts[status == 1]

    # 计算特征点的运动
    motion = np.sqrt(np.sum((good_new - good_old)**2, axis=1))
    motion_history.append(motion)


    # Draw tracking lines
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        frame = cv2.line(frame, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    # Display the result
    cv2.imshow('Optical Flow', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # 更新前一帧和特征点
    prev_frame = next_frame.copy()
    prev_pts = good_new.reshape(-1, 1, 2)


print(motion_history)
# # 可视化特征点的运动
# plt.figure(figsize=(10, 6))
# plt.plot(np.mean(motion_history, axis=1))
# plt.title('Camera Motion Over Time')
# plt.xlabel('Frame')
# plt.ylabel('Average Feature Point Motion')
# plt.show()