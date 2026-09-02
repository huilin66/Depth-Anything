import pandas as pd
import cv2

# # 打开视频文件
# video_path = r'E:\data\0111_testdata\record\V1\Visual\DJI_0844_W.MP4'
# video = cv2.VideoCapture(video_path)
#
# # 指定要提取的帧数（例如，提取第10帧）
# frame_number = 223
#
# # 设置视频的当前帧位置
# video.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
#
# # 读取指定帧
# success, frame = video.read()
#
# if success:
#     # 保存帧为图像文件
#     image_path = f"./frame_223.jpg"
#     cv2.imwrite(image_path, frame)
#     print(f"已保存第{frame_number}帧为图像文件：{image_path}")
# else:
#     print(f"无法读取第{frame_number}帧")
#
# # 关闭视频文件
# video.release()



# df = pd.read_csv('p1.csv', header=0, index_col=0)
# df['y_sum'] = df['y'].cumsum()
# # first_row_exceeding_3000 = df[df['y_sum'] > 3000].iloc[0]
# first_row_exceeding_3000 = df[df['y_sum'] > 2160].index[0]
# # print(df['y_sum'])
# print(first_row_exceeding_3000)