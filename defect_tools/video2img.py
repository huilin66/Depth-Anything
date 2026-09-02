import pandas as pd
import cv2, os
import numpy as np
from pathlib import Path
from tqdm import trange
state_csv = r'E:\repository\Depth-Anything\p1.csv'
video_path = r'E:\data\0111_testdata\record\V1\Visual\DJI_0844_W.MP4'
IMG_HEIGHT = 2160
IMG_WIDTH = 3840

# def get_framenum(state_csv):
#     df = pd.read_csv(state_csv, header=0, index_col=0)
#     df['y_sum'] = df['y'].cumsum()
#     df['x_sum'] = df['x'].cumsum()
#     max_length = df['y_sum'].max()
#     print(max_length)
#
#     frame_idxs = []
#     coords = []
#     for idx, img_gap in enumerate(range(0, int(max_length), IMG_HEIGHT)):
#
#         frame_idx = df[df['y_sum'] > img_gap].index[0]
#         print(idx, img_gap, frame_idx)
#         frame_idxs.append(frame_idx)
#         coord = df.loc[frame_idx].tolist()
#         coords.append(coord)
#
#     return frame_idxs, coords


def get_framenum(state_csv):
    df = pd.read_csv(state_csv, header=0, index_col=0)
    df['y_sum'] = df['y'].cumsum()
    df['x_sum'] = df['x'].cumsum()
    frame_idxs = [i for i in range(0, len(df)+1, 30)]
    coords = []
    for frame_idx in frame_idxs:
        coord = df.loc[frame_idx].tolist()
        coords.append(coord)
    return frame_idxs, coords

def get_frames(video_path, save_dir, frame_nums, coords):
    video = cv2.VideoCapture(video_path)

    for frame_num, coord in zip(frame_nums, coords):
        # 设置视频的当前帧位置
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)

        # 读取指定帧
        success, frame = video.read()

        image_path = os.path.join(save_dir, '%06d_%.4f_%.4f.png'%(frame_num, coord[-1], coord[-2]))
        cv2.imwrite(image_path, frame)

    video.release()


# def img_merge(save_dir):
#     pass
#     frame_list = os.listdir(save_dir)
#
#     df_info = pd.DataFrame(None, columns=['idx', 'x', 'y'])
#     for frame_name in frame_list:
#         name_strs = Path(frame_name).stem.split('_')
#         name_nums = [float(name_str) for name_str in name_strs]
#         df_info.loc[len(df_info)] = name_nums
#     print(df_info)
#
#     merge_height = int(df_info['y'].max()) + IMG_HEIGHT
#
#     x_min, x_max = int(df_info['x'].min()), int(df_info['x'].max())
#     merge_width = IMG_WIDTH + abs(x_min) + abs(x_max)
#
#     img_merge = np.zeros((merge_height, merge_width, 3), dtype=np.uint8)
#     for idx in range(len(frame_list)):
#         img_path = os.path.join(save_dir, frame_list[idx])
#         img = cv2.imread(img_path)
#         x = 0 #int(df_info['x'][idx] + abs(x_min))
#         y = int(df_info['y'][idx])
#         img_merge[merge_height-y - IMG_HEIGHT : merge_height-y, x:x + IMG_WIDTH] = img
#
#     img_merge = cv2.resize(img_merge, dsize=(0, 0), fx=0.05, fy=0.05)
#     cv2.imwrite(os.path.join(os.path.dirname(save_dir), 'img_merge.png'), img_merge)


def img_merge(save_dir):
    pass
    frame_list = os.listdir(save_dir)

    df_info = pd.DataFrame(None, columns=['idx', 'x', 'y'])
    for frame_name in frame_list:
        name_strs = Path(frame_name).stem.split('_')[:3]
        name_nums = [float(name_str) for name_str in name_strs]
        df_info.loc[len(df_info)] = name_nums
    print(df_info)

    merge_height = int(df_info['y'].max()) + IMG_HEIGHT

    x_min, x_max = int(df_info['x'].min()), int(df_info['x'].max())
    merge_width = IMG_WIDTH + abs(x_min) + abs(x_max)


    img_merge = np.zeros((merge_height, merge_width, 3), dtype=np.uint8)
    x = int(x_max - df_info['x'][0])
    img_merge[merge_height-IMG_HEIGHT:merge_height, x:x + IMG_WIDTH] = cv2.imread(os.path.join(save_dir, frame_list[0]))
    start_y, end_y = 0, 0
    for idx in trange(1, len(frame_list)):
        img_path = os.path.join(save_dir, frame_list[idx])
        img = cv2.imread(img_path)
        x = int(x_max - df_info['x'][idx])
        end_y = int(df_info['y'][idx])
        img_merge[merge_height-end_y-IMG_HEIGHT : merge_height-start_y-IMG_HEIGHT, x:x + IMG_WIDTH] = img[0:end_y-start_y]
        start_y = end_y

    # img_merge = cv2.resize(img_merge, dsize=(0, 0), fx=0.05, fy=0.05)
    cv2.imwrite(os.path.join(os.path.dirname(save_dir), 'img_merge_rgb.png'), img_merge)

if __name__ == '__main__':
    pass
    # frame_idxs, coords = get_framenum(state_csv)
    # get_frames(video_path,
    #            save_dir=r'E:\repository\Depth-Anything\defect_tools\imgs',
    #            frame_nums=frame_idxs, coords=coords)

    img_merge(save_dir=r'E:\repository\Depth-Anything\defect_tools\imgs',)
    # img_merge(save_dir=r'E:\repository\Depth-Anything\defect_tools\imgs_depth',)