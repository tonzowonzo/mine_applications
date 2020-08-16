import rasterio
import numpy as np
import keras
import matplotlib.pyplot as plt
import keras
from keras.models import load_model, Model
from numba import jit


input_path = "enmap/ENMAP01-____L2A-DT000326721_20170626T102020Z_001_V000204_20200406T201930Z-SPECTRAL_IMAGE.tif"
output_path = "output.tif"

def compress_enmap_image(input_path, output_path):
    """
    Compresses an enmap image with 208 bands to one of 3 bands.
    Model used is an autoencoder where the encoder is a 1D CNN and
    the decoder is an ANN.
    
    inputs:
        input_path: string, a path to the input enmap image.
        output_path: string, path to save the bottleneck image to.
    
    """
    with rasterio.open(input_path) as src:
        img = src.read()
        kwargs = src.profile
        img = np.moveaxis(img, 0, 2)
       
    model = load_model("compression_ae_3bottleneck.h5")
    
    x, y, z = img.shape
    
    @jit(nopython=True, fastmath=True)
    def prepare_data(img):
        img = (img / 10000).astype(np.float32)
        return img
    
    # Convert to reflectance.
    img = prepare_data(img)
    
    # Get rid of null values as well as oversaturated pixels.
    img[img <= 0] = 0
    img[img >= 1] = 1
    img = img.reshape(x * y, z, 1)
    layer_name = 'bottleneck'
    intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(img, batch_size=512)
    
    intermediate_output = intermediate_output.reshape((x, y, 3))
    
    kwargs["count"] = 3
    kwargs["dtype"] = "float32"
    
    plt.figure(figsize=(12, 12))
    plt.imshow(intermediate_output)
    plt.show()

    intermediate_output = np.moveaxis(intermediate_output, 2, 0)
    with rasterio.open(output_path, "w", **kwargs) as dst:
        dst.write(intermediate_output)
        
if __name__ == "__main__":
    compress_enmap_image(input_path, output_path)
    
