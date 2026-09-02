import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm

dep_img_path = r'metric_depth/my_test/output/DJI_0951_W_depth.png'
rgb_img_path = r'metric_depth/my_test/input/DJI_0951_W.JPG'


# Depth camera parameters:
FX_DEPTH = 5908
FY_DEPTH = 5908

# FX_DEPTH = 5485.714285714285
# FY_DEPTH = 5485.714285714285
CX_DEPTH = 4000
CY_DEPTH = 3000
Z_dist = 300
FX_RGB = FX_DEPTH
FY_RGB = FY_DEPTH
CX_RGB = CX_DEPTH-4000
CY_RGB = CY_DEPTH

def img2pcd():
# Read depth image:
    depth_image = iio.imread()
    rgb_image = iio.imread()[::-1, ...]#[:, ::-1]


    if len(depth_image.shape) == 3:
        depth_image = depth_image[:, :, 0]
    # print properties:
    print(f"Image resolution: {depth_image.shape}")
    print(f"Data type: {depth_image.dtype}")
    print(f"Min value: {np.min(depth_image)}")
    print(f"Max value: {np.max(depth_image)}")



    # compute point cloud:
    pcd = []
    # height, width = depth_image.shape
    # for i in tqdm(range(height)):
    #     for j in range(width):
    #         z = depth_image[i][j] + 500
    #         x = (j - CX_DEPTH) * z / FX_DEPTH
    #         y = (i - CY_DEPTH) * z / FY_DEPTH
    #         pcd.append([x, y, z])
    # get depth resolution:

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

    R = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    T = np.array([0, 0, 0])

    # compute indices:
    jj = np.tile(range(width), height)
    ii = np.repeat(range(height), width)

    # Compute constants:
    xx = (jj - CX_DEPTH) / FX_DEPTH
    yy = (ii - CY_DEPTH) / FY_DEPTH

    # transform depth image to vector of z:
    length = height * width
    z = depth_image.reshape(length)+Z_dist

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
