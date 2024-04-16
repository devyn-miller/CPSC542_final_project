import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ImageAugmenter:
    def __init__(self, IMG_WIDTH, IMG_HEIGHT):
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_HEIGHT = IMG_HEIGHT


    def augment(self):
        '''Applies augmentation to the images.'''
        
        return ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,       # Random rotations from 0 to 40 degrees
            width_shift_range=0.2,   # Random horizontal shifts
            height_shift_range=0.2,  # Random vertical shifts
            shear_range=0.2,         # Shear transformations
            zoom_range=0.2,          # Random zoom
            horizontal_flip=True,    # Enable horizontal flipping
            fill_mode='nearest'      # Strategy for filling in newly created pixels
        )

