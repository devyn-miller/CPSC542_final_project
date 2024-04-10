import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ImageAugmenter:
    def __init__(self, image_width, image_height):
        self.image_width = image_width
        self.image_height = image_height

    def augment(self, bw_image, colored_image):
        '''The main method that augments the images. 

            '''
            
        data_gen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='constant')
        
        bw_image = bw_image.reshape((1,) + bw_image.shape)
        colored_image = colored_image.reshape((1,) + colored_image.shape)
        
        bw_image = data_gen.flow(bw_image, batch_size=1)
        colored_image = data_gen.flow(colored_image, batch_size=1)

        return bw_image, colored_image
