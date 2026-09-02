import cv2
video_path = r'E:\data\0111_testdata\record\V1\Visual\DJI_0844_W.MP4'
# 读取视频
cap = cv2.VideoCapture(video_path)

# 创建Lucas-Kanade光流对象
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 读取第一帧
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    # 读取下一帧
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 计算光流
    next_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, None, None, **lk_params)

    # 计算相机运动
    dx = next_points[:, 0, 0] - prev_points[:, 0, 0]
    dy = next_points[:, 0, 1] - prev_points[:, 0, 1]

    # 更新前一帧的信息
    prev_gray = gray.copy()
    prev_points = next_points.copy()

    # 在图像上绘制运动向量
    for i in range(len(dx)):
        cv2.arrowedLine(frame, (int(prev_points[i, 0, 0]), int(prev_points[i, 0, 1])),
                        (int(prev_points[i, 0, 0] + dx[i]), int(prev_points[i, 0, 1] + dy[i])),
                        (0, 255, 0), 2)

    cv2.imshow('Motion Estimation', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
