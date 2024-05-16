import os
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Concatenate, Lambda, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from skimage.color import rgb2lab, lab2rgb, gray2rgb

class VGG16_transfer:
    def __init__(self, image_shape=(480,360,1)):
        self.input_shape = (image_shape[0], image_shape[1])
        self.encoder = None
        self.decoder = None
        self.model = None

    def build_model(self, hp):
        model_input = (self.input_shape + (3,))
        print("HERE")
        print(model_input)
        model_output = (self.input_shape + (2,))

        # Load the VGG16 model pre-trained on ImageNet
        # Use model for encoder, expects rgb input
        print("building vgg16 encoder")
        self.encoder: Model = VGG16(weights='imagenet', include_top=False, input_shape=model_input)

        # Freeze the VGG16 layers
        for layer in self.encoder.layers:
            layer.trainable = False

        # Access the output shape of the encoder
        encoder_output_shape = self.encoder.output_shape[1:] # Drop batch size element

        # Build the decoder model
        print("Building sequential decoder...")
        self.decoder = self.build_decoder(hp, encoder_output_shape)
        
        
        self.model = Model(inputs=self.encoder.input, outputs=self.decoder)

        
        print("Model built! Ready to train.")
        return self.model
        
    def build_decoder(self, hp, encoder_output_shape):
        num_blocks = hp.Int('num_blocks', min_value=1, max_value=5, step=1)
        initial_num_filters = hp.Int('initial_num_filters', min_value=64, max_value=128, step=16)
        conv_layers = hp.Int('conv_layers', min_value=1, max_value=2, step=1)
        #scaling = hp.Int('scaling', min_value=-1, max_value=2, step=1)
        
        inputs = Input(shape=encoder_output_shape)
        decoder = Conv2D(initial_num_filters, (3,3), activation='relu', padding='same')(inputs)
        
        for i in range(num_blocks):
            num_filters = initial_num_filters * (2 ** (i))
            # Hyperparameter
            
            for i in range(conv_layers):
                decoder = self.conv_block(decoder, num_filters, 3)
            decoder = UpSampling2D((2, 2))(decoder)
            print("here")
        
        #decoder = Lambda(lambda x: tf.image.resize(x, self.input_shape, method=tf.image.ResizeMethod.BILINEAR))(decoder)
        decoder = ResizeLayer(self.input_shape[:2])(decoder)

        
        
        return decoder
    
    def conv_block(self, input_tensor, num_filters, filter_size):
        x = Conv2D(num_filters, (filter_size, filter_size), padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        return x
    
class ResizeLayer(tf.keras.layers.Layer):
    def __init__(self, target_size, **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.target_size = target_size

    def call(self, inputs):
        return tf.image.resize(inputs, self.target_size)