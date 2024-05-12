import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm
import cv2


class ListImageAugmenter:
    def __init__(self, IMG_WIDTH, IMG_HEIGHT):
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_HEIGHT = IMG_HEIGHT
        self.datagen = ImageDataGenerator(
            rotation_range=40,       # Random rotations from 0 to 40 degrees
            width_shift_range=0.2,   # Random horizontal shifts
            height_shift_range=0.2,  # Random vertical shifts
            shear_range=0.2,         # Shear transformations
            zoom_range=0.2,          # Random zoom
            horizontal_flip=True,    # Enable horizontal flipping
            fill_mode='nearest'      # Strategy for filling in newly created pixels
        )

    def augment_images(self, image_list):
        '''Applies augmentation to a list of images and returns the augmented images.'''
        augmented_images = []
        for img in image_list:
            img = img.reshape((1,) + img.shape)  # Reshape to (1, height, width, channels)
            i = 0
            for batch in self.datagen.flow(img, batch_size=1):
                augmented_images.append(batch[0])
                i += 1
                if i > 0:  # Stop after generating one batch of augmented images
                    break
        return np.array(augmented_images)
    
    def load_images(self, path):
        images = []
        for filename in tqdm(os.listdir(path)):
            file_path = os.path.join(path, filename)
            image = load_img(file_path, target_size=(self.IMG_WIDTH, self.IMG_HEIGHT), color_mode='rgb')
            image = img_to_array(image)
            images.append(image)
        return np.array(images)
    
    def normalize_images(self, image_list):
        '''Normalizes a list of images by dividing pixel values by 255.0.'''
        normalized_images = []
        for img in image_list:
            normalized_images.append(img / 255.0)
        return np.array(normalized_images)

    def rgb_to_black_and_white_cv2(self, rgb_images):
        black_and_white_images = []
        for img in rgb_images:
            # OpenCV expects the image in BGR format
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            grayscale_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            black_and_white_images.append(grayscale_image)
        return np.array(black_and_white_images)
