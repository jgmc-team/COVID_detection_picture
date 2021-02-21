# %%
#  !pip install -q pyyaml h5py
# !pip install https://github.com/ipython-contrib/jupyter_contrib_nbextensions/tarball/master
# !pip install jupyter_nbextensions_configurator
# !jupyter contrib nbextension install --user
# !jupyter nbextensions_configurator enable --user

# %%
import datetime as dt
# t_start = dt.datetime.now()

# %%
try:
    from google.colab import drive
    drive.mount('/content/drive')
    ABS_PATH = '/content/drive/MyDrive/JN/NN/COVID_detection_picture'
except ModuleNotFoundError:
    from os import getcwd
    ABS_PATH = getcwd()

# %%
import os
import gzip
import numba
import zipfile

import nibabel as nib
from scipy import ndimage
import scipy.special as sc

import numpy as np
from numba import jit
from numba import njit
import threading as tr

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


### numba warrnings ###########
from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# %%
# def get_model_8L_(width=128, height=128, depth=64):
#     """Build a 3D convolutional neural network model."""

#     inputs = keras.Input((width, height, depth, 1))

#     x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
#     x = layers.MaxPool3D(pool_size=2)(x)
#     x = layers.BatchNormalization(center=True, scale=True)(x)
#     

#     x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
#     x = layers.MaxPool3D(pool_size=2)(x)
#     x = layers.BatchNormalization(center=True, scale=True)(x)
#    


#     ### add 64N layer ##################################################
#     x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
#     x = layers.MaxPool3D(pool_size=2)(x)
#     x = layers.BatchNormalization(center=True, scale=True)(x)
#  

#     x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
#     x = layers.MaxPool3D(pool_size=1)(x)
#     x = layers.BatchNormalization(center=True, scale=True)(x)
#    

#     x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
#     x = layers.MaxPool3D(pool_size=2)(x)
#     x = layers.BatchNormalization()(x)

#     x = layers.GlobalAveragePooling3D()(x)
#     x = layers.Dense(units=512, activation="relu")(x)
#     x = layers.Dropout(0.3)(x)

#     outputs = layers.Dense(units=1, activation="sigmoid")(x)

#     # Define the model.
#     model = keras.Model(inputs, outputs, name="3dcnn")
#     return model

# %%
# @tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),))
def create_model_():
    # Loads the weights
    loaded_model = keras.models.load_model(ABS_PATH + "/Keras/model/model_keras_F1_0819.h5")
    # model.load_weights(checkpoint_path)
    return loaded_model
    # model.summary()
loaded_model = create_model_()

# %%
def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan
def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

# @jit(parallel=True, cache=True, )
def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    # desired_depth = 64 #old
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    # img = ndimage.rotate(img, 90, reshape=False)
    img = np.rot90(img)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume
### TESTING BLOCK #####################
# test_image = process_scan(test_image)
# test_image.shape
######################################

# %%
def main(path):
    label_image = ['normal', 'abnormal']
    image = process_scan(path)
    prediction = loaded_model.predict(np.expand_dims(image, axis=0))[0];
    scores = [1 - prediction[0], prediction[0]]
    if scores[0] > scores[1]:
        return [label_image[0], str(scores[0])]
    else:
        return [label_image[1], str(scores[1])]


# %%
### TEST MODEL # AFTER TESTING COMMENT THIS BLOCK ###
# print(os.getcwd()) # check path to project folder
# path = os.getcwd() + "/MRI/test_MRI.nii" # full path to image.nii 
# path = os.getcwd() + "/MRI/sick_7ff18f5d3de11b9ae7a9e5d651313fbd.nii" # check classification
# print(path) # check full path to image .nii
# main(path)
####################################################

# %%
### THIS BLOCK VIEWING IMAGE FOR VISUAL CONTROLL ##########################
# test_image = path + lst_images[0]
# prediction = loaded_model.predict(np.expand_dims(test_image, axis=0))[0]
# scores = [1 - prediction[0], prediction[0]]
def plot_result(path):
    image = process_scan(path)
    prediction = create_model_().predict(np.expand_dims(image, axis=0))[0];
    scores = [1 - prediction[0], prediction[0]]
    scores_lst = []
    for score, name in zip(scores, class_names):
        scores_lst.append([score, name])
    if scores_lst[0][0] > scores_lst[1][0]:
        print(f"MRI is {100 * scores_lst[0][0]:.2f}% is: {scores_lst[0][1]} | Dim.:", image.shape)
    else:    
        print(f"MRI is {100 * scores_lst[1][0]:.2f}% is: {scores_lst[1][1]} | Dim.:", image.shape)
    plt.imshow(np.squeeze(image[:, :, 2]), cmap="gray")
    plt.show()

# %%
# t_finish = dt.datetime.now() - t_start
# t_finish.seconds

# %%
main(ABS_PATH + "/MRI/sick_7c7160149aec1ebf15b28166f5458c49.nii")

# %%
#!ipynb-py-convert K8L_detect.ipynb K8L_detect.py

# %%
