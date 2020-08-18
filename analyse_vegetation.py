import rasterio
import matplotlib.pyplot as plt
import os
import numpy as np

os.chdir("C:/Users/Tim/Desktop/Myna/mine_applications/vegetation_surveys")
import vegetation_indices_s2
from vegetation_indices_s2 import ndvi

def plot_image(img):
    plt.figure(figsize=(14, 14))
    plt.imshow(img)
    plt.show()

def create_rgb(img):
    rgb = np.zeros((img.shape[0], img.shape[1], 3))
    rgb[:, :, 0] = img[:, :, 2]
    rgb[:, :, 1] = img[:, :, 1]
    rgb[:, :, 2] = img[:, :, 0]
    rgb = rgb * 3
    return rgb

def read_image(img_path):
    with rasterio.open(img_path) as src:
        img = src.read()
        kwargs = src.profile
        img = np.moveaxis(img, 0, 2)
   
    return img, kwargs

def analyse_data(s2_img_path):
    img, kwargs = read_image(s2_img_path)
    
    img = (img / 10000).astype(np.float32)
    rgb = create_rgb(img)
    plot_image(rgb)
    
    ndvi_img = vegetation_indices_s2.ndvi(img).astype(np.float64)
    ndvi_img[np.isnan(ndvi_img)] = -1
    plot_image(ndvi_img)
    
    ndwi_img = vegetation_indices_s2.ndwi(img).astype(np.float64)
    ndwi_img[np.isnan(ndwi_img)] = -1
    plot_image(ndwi_img)
    
    hmssi_img = vegetation_indices_s2.hmssi(img).astype(np.float64)
    hmssi_img[np.isnan(hmssi_img)] = 0
    hmssi_img[hmssi_img < 0] = 0
    hmssi_img[hmssi_img > 20] = 20
    hmssi_img = vegetation_indices_s2.scale_img(hmssi_img)
    plot_image(hmssi_img)
    
    stack = np.dstack([ndvi_img, ndwi_img, hmssi_img])
    plot_image(stack)

    return stack
stack_2019 = analyse_data("example_data/mt_henry_s2_2019.tif")
stack_2020 = analyse_data("example_data/mt_henry_s2_2020.tif")

diff = stack_2019 - stack_2020
diff_hmssi = stack_2019[:, :, 2] - stack_2020[:, :, 2]
plot_image(diff)
plot_image(rgb)

plot_image(diff_hmssi)

diff_hmssi = diff_hmssi.astype(np.float32)
kwargs["count"] = 1
kwargs["dtype"] = "float32"
with rasterio.open("test_hmssi_diff.tif", "w", **kwargs) as dst:
    dst.write(diff_hmssi, 1)
