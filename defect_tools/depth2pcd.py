import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm

FX_DEPTH = 5908
FY_DEPTH = 5908

CX_DEPTH = 4000
CY_DEPTH = 3000
Z_dist = 300
FX_RGB = FX_DEPTH
FY_RGB = FY_DEPTH
CX_RGB = CX_DEPTH-4000
CY_RGB = CY_DEPTH

# CX_DEPTH = 3840
# CY_DEPTH = 2160
# Z_dist = 300
# FX_RGB = FX_DEPTH
# FY_RGB = FY_DEPTH
# CX_RGB = CX_DEPTH-2000
# CY_RGB = CY_DEPTH

# CX_DEPTH = 4161
# CY_DEPTH = 40107
# Z_dist = 300
# FX_RGB = FX_DEPTH
# FY_RGB = FY_DEPTH
# CX_RGB = CX_DEPTH-2000
# CY_RGB = CY_DEPTH

R = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
])
T = np.array([0, 0, 0])


def depth2pcd_gray(depth_img_path):
    depth_image = iio.imread(depth_img_path)[::-1, ...]
    if len(depth_image.shape) == 3:
        depth_image = depth_image[:, :, 0]

    height, width = depth_image.shape
    # compute indices:
    jj = np.tile(range(width), height)
    ii = np.repeat(range(height), width)
    # Compute constants:
    xx = (jj - CX_DEPTH) / FX_DEPTH
    yy = (ii - CY_DEPTH) / FY_DEPTH
    # transform depth image to vector of z:
    length = height * width
    z = depth_image.reshape(height * width) + Z_dist
    # compute point cloud
    pcd = np.dstack((xx * z, yy * z, z)).reshape((length, 3))

    pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)  # set pcd_np as the point cloud points
    # Visualize:
    o3d.visualization.draw_geometries([pcd_o3d])

def depth2pcd_color(depth_img_path, rgb_img_path):
    depth_image = iio.imread(depth_img_path)[::-1, ...]
    rgb_image = iio.imread(rgb_img_path)[::-1, ...]  # [:, ::-1]
    if len(depth_image.shape) == 3:
        depth_image = depth_image[:, :, 0]

    height, width = depth_image.shape
    # compute indices:
    jj = np.tile(range(width), height)
    ii = np.repeat(range(height), width)
    # Compute constants:
    xx = (jj - CX_DEPTH) / FX_DEPTH
    yy = (ii - CY_DEPTH) / FY_DEPTH
    # transform depth image to vector of z:
    length = height * width
    z = depth_image.reshape(height * width) + Z_dist
    # compute point cloud
    pcd = np.dstack((xx * z, yy * z, z)).reshape((length, 3))
    cam_RGB = np.apply_along_axis(np.linalg.inv(R).dot, 1, pcd) - np.linalg.inv(R).dot(T)
    xx_rgb = ((cam_RGB[:, 0] * FX_RGB) / cam_RGB[:, 2] + CX_RGB + width / 2).astype(int).clip(0, width - 1)
    yy_rgb = ((cam_RGB[:, 1] * FY_RGB) / cam_RGB[:, 2] + CY_RGB).astype(int).clip(0, height - 1)
    colors = rgb_image[yy_rgb, xx_rgb] / 255

    # Convert to Open3D.PointCLoud:
    pcd_o3d = o3d.geometry.PointCloud()  # create a point cloud object
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
    # Visualize:
    o3d.visualization.draw_geometries([pcd_o3d])

if __name__ == '__main__':
    depth_path = r'E:\repository\Depth-Anything\metric_depth\my_test\output\DJI_0951_W_depth.png'
    rgb_path = r'E:\repository\Depth-Anything\metric_depth\my_test\input\DJI_0951_W.JPG'
    # depth_path = r'E:\repository\Depth-Anything\defect_tools\imgs_depth\000000_0.0182_0.0060_depth.png'
    # rgb_path = r'E:\repository\Depth-Anything\defect_tools\imgs\000000_0.0182_0.0060.png'

    # depth_path = r'E:\repository\Depth-Anything\defect_tools\img_merge_depth.png'
    # rgb_path = r'E:\repository\Depth-Anything\defect_tools\img_merge_rgb.png'



    # depth2pcd_gray(depth_path)

    depth2pcd_color(depth_path, rgb_path)

