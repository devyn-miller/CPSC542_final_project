import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ImageAugmenter:
    def __init__(self, image_width, image_height):
        self.image_width = image_width
        self.image_height = image_height

    def augment(self, bw_image, colored_image):
        '''Applies augmentation to the images.'''
        data_gen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        # Assuming bw_image and colored_image are numpy arrays
        # Reshape the images to add the batch dimension
        bw_image = bw_image.reshape((1,) + bw_image.shape)
        colored_image = colored_image.reshape((1,) + colored_image.shape)

        # Apply augmentation to the black and white image
        bw_image_gen = data_gen.flow(bw_image, batch_size=1)
        # Apply augmentation to the colored image
        colored_image_gen = data_gen.flow(colored_image, batch_size=1)

        # Return the augmented images
        return next(bw_image_gen)[0], next(colored_image_gen)[0]
