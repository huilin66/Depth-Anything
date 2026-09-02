import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd

video_path = r'E:\data\0111_testdata\record\V1\Visual\DJI_0844_W.MP4'
feature_points = 100


# 参数
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict(maxCorners=feature_points, qualityLevel=0.3, minDistance=7, blockSize=7)

# 视频源
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


# Shi-Tomasi角点检测
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# 创建遮罩，避开水印区域
mask_select = np.zeros_like(old_gray)
mask_select[int(old_gray.shape[0]*0.05):int(old_gray.shape[0]*0.25),
     int(old_gray.shape[1]*0.25):int(old_gray.shape[1]*0.75)] = 1

p0 = cv2.goodFeaturesToTrack(old_gray, mask=mask_select, **feature_params)

# 创建一个mask
mask = np.zeros_like(old_frame)

# 创建一个空列表来保存运动参数
motion_params = [[], []]

frame_id = 0
while True:
    print('%d:%d'%(frame_id, frame_count))
    frame_id += 1
    # if frame_id>200:
    #     break
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 计算光流
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is None:
        print()
    # 选择好的点
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    dys = good_new[:, 1] - good_old[:, 1]
    mask_y = dys>=0

    good_new = good_new[mask_y]
    good_old = good_old[mask_y]
    dys = dys[mask_y]
    dxs = good_new[:, 0] - good_old[:, 0]
    dx = np.mean(dxs)
    dy = np.mean(dys)
    motion_params[0].append(dx)
    motion_params[1].append(dy)

    # # Estimate homography
    # H, _ = cv2.findHomography(good_old, good_new, cv2.RANSAC, 5.0)
    #
    # # Use homography to warp the old points
    # good_old_warped = cv2.perspectiveTransform(good_old.reshape(-1, 1, 2), H)
    #
    # # Calculate and save average displacement
    # dx = np.mean(good_old_warped[:, 0, 0] - good_new[:, 0])
    # dy = np.mean(good_old_warped[:, 0, 1] - good_new[:, 1])
    # motion_params.append((dx, dy))


    if len(good_new)<int(feature_points*0.5):
        # print('add %d -> 100'%len(p0))
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=mask_select, **feature_params)
    else:
        p0 = good_new.reshape(-1, 1, 2)

    # 绘制轨迹
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = map(int, new.ravel())
        c, d = map(int, old.ravel())
        mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
        frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)
    img = cv2.add(frame, mask)

    cv2.imshow('frame', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # 更新上一帧和上一点
    old_gray = frame_gray.copy()
    # p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()

df = pd.DataFrame(None, columns=None)
df['x'] = motion_params[0]
df['y'] = motion_params[1]
df['x'] = df['x'].interpolate()
df['y'] = df['y'].interpolate()
# print(df)

df.to_csv('p1.csv')
# for i, (dx, dy) in enumerate(motion_params):
#     print(f'Frame {i} -> Frame {i+1}: dx = {dx}, dy = {dy}')