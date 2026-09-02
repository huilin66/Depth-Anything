import cv2
import numpy as np

# 读取两幅图像
image1 = cv2.imread(r"E:\repository\defectdet\swa_script\frames\v1_6F.jpg", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(r"E:\repository\defectdet\swa_script\frames\v1_7F.jpg", cv2.IMREAD_GRAYSCALE)

# 初始化ORB特征点检测器
orb = cv2.ORB_create()

# 检测特征点和计算描述子
keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

# 创建BFMatcher对象，进行特征点匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# 选择最佳匹配
matches = sorted(matches, key=lambda x: x.distance)

# 计算平移向量
translation_vector = np.zeros((2, 1))
for match in matches:
    point1 = keypoints1[match.queryIdx].pt
    point2 = keypoints2[match.trainIdx].pt
    translation_vector += np.array(point2) - np.array(point1)

# 计算平均平移向量
average_translation = translation_vector / len(matches)

print(f"平移向量：{average_translation[0][0]} 像素（x方向），{average_translation[1][0]} 像素（y方向）")
