import rasterio
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization, Convolution1D, MaxPooling1D, Flatten
os.chdir(r"C:/Users/Tim/Desktop/Myna/mine_applications/hyperspectral_compression/")


X = np.zeros((0, 218)).astype(np.int16)



for img_name in os.listdir("enmap"):
    
    with rasterio.open(f"enmap/{img_name}") as src:
        img = src.read()
        img = np.moveaxis(img, 0, 2)
        img = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
        img = img[::4, :]
        X = np.append(X, img, axis=0)
        del img
        
        
X = (X / 10000).astype(np.float32)
X[X <= 0] = 0
X[X > 1] = 1
X = X[X[:, 0] != 0]



X = X.reshape(X.shape[0], X.shape[1], 1)
y = X.reshape(X.shape[0], X.shape[1])

model = Sequential()
model.add(Convolution1D(64, 2, padding="same", input_shape =(218,1)))
model.add(Convolution1D(64, 2, padding="same"))
model.add(MaxPooling1D())
model.add(Dropout(0.2))

model.add(Convolution1D(32, 2, padding="same"))
model.add(Convolution1D(32, 2, padding="same"))
model.add(MaxPooling1D())
model.add(Dropout(0.2))

model.add(Convolution1D(16, 2, padding="same"))
model.add(Convolution1D(16, 2, padding="same"))
model.add(MaxPooling1D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(3))
model.add(LeakyReLU())
model.add(BatchNormalization(name="bottleneck"))
model.add(Dense(16, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dense(218, activation="sigmoid"))

model.summary()

model.compile(optimizer="adam", loss="mse")

model.fit(X, y, batch_size=512, validation_split=0.1, epochs=10)
model.save("compression_ae_3bottleneck.h5")

del X
del img

with rasterio.open(f"enmap/{img_name}") as src:
    img = src.read()
    kwargs = src.profile
    img = np.moveaxis(img, 0, 2)
    img = (img / 10000).astype(np.float32)
    img[img <= 0] = 0
    img[img >= 1] = 1
    x, y, z = img.shape

    img = img.reshape(x * y, z, 1)
    
    layer_name = 'bottleneck'
    intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(img, batch_size=512)
    
    intermediate_output = intermediate_output.reshape((x, y, 3))
    

kwargs["count"] = 3
kwargs["dtype"] = "float32"

intermediate_output = np.moveaxis(intermediate_output, 2, 0)
with rasterio.open("test.tif", "w", **kwargs) as dst:
    dst.write(intermediate_output)
       

pred = model.predict(img, batch_size=512)
pred = pred.reshape((x, y, z))

kwargs["count"] = 218
kwargs["dtype"] = "float32"

pred = np.moveaxis(pred, 2, 0)
with rasterio.open("test_output.tif", "w", **kwargs) as dst:
    dst.write(pred)
