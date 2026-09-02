import matplotlib.pyplot as plt
from skimage import io

img = io.imread(r'E:\repository\Depth-Anything\defect_tools\imgs_depth\000000_0.0182_0.0060_depth.png')
plt.imshow(img)
plt.show()