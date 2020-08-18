# Import libraries.
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
"""
Calculate vegetation indices from a stack of S2 images where containing bands:
    [2, 3, 4, 5, 6, 7, 8, 11, 12, 8A]
    [0, 1, 2, 3, 4, 5, 6, 7,  8,  9]
"""


    
def plot_image(img):
    plt.figure(figsize=(14, 14))
    plt.imshow(img)
    plt.show()

def red_edge1(img):
    re1 = img[:, :, 3] / img[:, :, 2]
    return re1


def red_edge2(img):
    re2 = (img[:, :, 3] - img[:, :, 2]) / (img[:, :, 3] + img[:, :, 2])
    return re2


def ndvi(img):
    ndvi = (img[:, :, 6] - img[:, :, 2]) / (img[:, :, 6] + img[:, :, 2])
    return ndvi


def savi(img, L):
    """
    Where:
        L: float, soil brightness factor, ranges from 0 to 1.
    """
    savi = (img[:, :, 6] - img[:, :, 2]) / (img[:, :, 6] + img[:, :, 2] + L) * (1.0 + L)
    return savi


def msi(img):
    """
    Moisture index.
    """
    msi = img[:, :, 7] / img[:, :, 6]
    return msi


def lwci(img):
    """
    Leaf water content.
    """
    lwci = (np.log(1 - (img[:, :, 6] - img[:, :, 7]))) / (-np.log(1 - (img[:, :, 6] - img[:, :, 7])))
    return lwci


def arvi(img, y):
    """
    Atmospherically resistant vegetation index.
    """
    arvi = (img[:, :, 6] - img[:, :, 2]) - (y * (img[:, :, 2] - img[:, :, 0]))
    return arvi

    
def evi(img):
    """
    Enhanced vegetation index.
    """
    evi = ((img[:, :, 6] - img[:, :, 2])) / ((img[:, :, 6] + (6*img[:, :, 2]) + (7.5*img[:, :, 0]) + 1))
    return evi


def tvi(img):
    """
    Transformed vegetation index.
    """
    tvi = 0.5 * (( 120 * (img[:, :, 4] - img[:, :, 1])) - (200 * (img[:, :, 2] - img[:, :, 1])))
    return tvi


def ci_red_edge(img):
    """
    Red-edge chlorophyll index.
    """
    cire = (img[:, :, 5] / img[:, :, 3]) - 1
    return cire

       
def psri(img):
    """
    Plant senesense reflection index.
    """
    psri = (img[:, :, 2] - img[:, :, 0]) / img[:, :, 4]
    return psri

def hmssi(img):
    """
    Heavy metal stress sensitive index.
    """
    hmssi = ci_red_edge(img) / psri(img)
    return hmssi

def ndwi(img):
    """
    Normalized difference water index.
    """
    ndwi = (img[:, :, 6] - img[:, :, 7]) / (img[:, :, 6] + img[:, :, 7])
    return ndwi
    
def scale_img(img):
    """
    Scales the image between 0 and 1.
    """
    scaled_img = (img - img.min()) / (img.max() - img.min())
    return scaled_img



if __name__ == "__main__":
    
    with rasterio.open(r"C:/Users/Tim/Desktop/Myna/mine_applications/vegetation_surveys/s2/example1.tif") as src:
        img = src.read()
        img = np.moveaxis(img, 0, 2)
        img = (img / 10000).astype(np.float32)
        
    hmssi_img = hmssi(img)
    hmssi_img = np.nan_to_num(hmssi_img)
    hmssi_img = scale_img(hmssi_img)
    plot_image(hmssi_img)
    
    evi_img = evi(img)
    evi_img[evi_img >= 0.3] = 0.3
    evi_img = np.nan_to_num(evi_img)
    evi_img = scale_img(evi_img)
    plot_image(evi_img)
    
    tvi_img = tvi(img)
    tvi_img = np.nan_to_num(tvi_img)
    tvi_img = scale_img(tvi_img)
    plot_image(tvi_img)