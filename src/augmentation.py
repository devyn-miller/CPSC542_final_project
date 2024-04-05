import tensorflow as tf

class ImageAugmenter:
    def __init__(self, image_width, image_height):
        self.image_width = image_width
        self.image_height = image_height

    def augment(self, bw_image, colored_image):
        '''The main method that augments the images.  Here are some fun ones:
        data_gen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='constant')
            if tf.random.uniform(()) > 0.5:
                image = tf.image.flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.2)

            '''
        


        return bw_image, colored_image
