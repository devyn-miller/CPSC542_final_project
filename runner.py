# %% [markdown]
# ## Auto-Encoder using VGG16 Transfer Learning 
# Tyler Lewis
# 
# May 2024

# %%
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from PIL import Image
import cv2
from skimage.color import rgb2lab, lab2rgb, gray2rgb
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
# from skimage.transform import resize
# from skimage.io import imsave
# import keras
# import os


#Local
from src.preprocessing import dataset

# %%
# HYPERPARAMETERS

TRAIN_EPOCHS: int = 1
LEARNING_RATE = 0.001
IMAGE_SHAPE: tuple = (360,640)
BATCH_SIZE: int = 32

# Define helper functions

def combine_l_ab(l: np.array, ab: np.array) -> np.array:
    ab = ab*128
    cur = np.zeros((IMAGE_SHAPE + (3,)))
    
    if len(l.shape) == 2:
        cur[:,:,0] = l
    elif len(l.shape) == 3:
        cur[:,:,0] = l[:, :, 0]
    else:
        print('error in combining lab img')
    cur[:,:,1:] = ab

    return cur

def display_img(img, gray = False): 
    cmap = 'gray' if gray else None
    plt.imshow(img, cmap=cmap) 
    plt.axis('off')  # Hide the axis
    plt.show()

#
# Define encoder
#
encoder_input_shape = (IMAGE_SHAPE + (3,)) # Add color dim to input shape
encoder: Model = VGG16(include_top=False, weights='imagenet', input_shape=encoder_input_shape)
tf.keras.config.enable_unsafe_deserialization()
decoder = tf.keras.models.load_model('colorize_decoder_may7.h5')

# %%
#Loading data
import cv2

testpath = 'data/vibrant/rgb/12_rgb_261.jpeg'

testing = np.array(cv2.imread(testpath))

testing_lab = rgb2lab(testing)
l = testing_lab[:,:,0]
l_input = gray2rgb(l).reshape(((1,) + IMAGE_SHAPE + (3,)))

# %%
#Predicting using saved model.

import cv2

testpath = 'data/vibrant/rgb/12_rgb_261.jpeg'

testing = np.array(cv2.imread(testpath))

testing_lab = rgb2lab(testing)
l = testing_lab[:,:,0]
l_input = gray2rgb(l).reshape(((1,) + IMAGE_SHAPE + (3,)))


bottleneck = encoder.predict(l_input)

display_img(l, gray=True)


# %%
output = decoder.predict(bottleneck)

# %%
print(l.shape)
output[0].shape

# %%
cur = combine_l_ab(l,output[0])

display_img(lab2rgb(cur))


