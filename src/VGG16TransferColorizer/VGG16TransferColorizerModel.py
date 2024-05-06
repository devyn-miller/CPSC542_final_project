'''
Tyler Lewis May 2024
VGG16 Transfer Colorizer Model

This architecture follows the typical structure of an encoder-decoder model for image-to-image tasks, where the encoder (VGG16) extracts high-level features from the input image, and the decoder upsamples and reconstructs the output (in this case, the colorized image).

During training, the weights of the VGG16 encoder will remain frozen, and only the weights of the decoder will be updated based on the colorization task. This approach leverages the pre-trained weights of the VGG16 model, which has learned useful features for image recognition, and fine-tunes the decoder to learn the colorization mapping.

Note that you'll need to compile the self.model with appropriate loss functions and optimizers before training, and you may also want to add additional layers or modify the decoder architecture based on your specific requirements.
'''

import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Concatenate, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from skimage.color import rgb2lab, lab2rgb, gray2rgb

# from helper import *

class VGG16TransferColorizer:
    def __init__(self, image_shape=(720, 1080)):
        self.input_shape = image_shape
        self.is_trained = False
        self.model = None

    def predict(self, grayscale_image):
        if self.model is None:
            print("Tried to predict before model was built")
            exit()

        # Convert 1 channel grayscale to RGB (repeat across 3 channels)
        input_l = gray2rgb(grayscale_image).reshape(((1,) + self.input_shape + (3,)))
        # Predict ab channel using model
        output_ab = self.model.predict(input_l)

        return output_ab

    def build_model(self):
        model_input = (self.input_shape + (3,))
        model_output = (self.input_shape + (2,))

        # Build the VGG16 encoder
        print("Building VGG16 encoder...")
        encoder = VGG16(weights='imagenet', include_top=False, input_shape=model_input)

        # Freeze the VGG16 layers
        for layer in encoder.layers:
            layer.trainable = False

        # Build the decoder model
        print("Building sequential decoder...")
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder.output)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(2, (3, 3), activation='tanh', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Lambda(lambda x: tf.image.resize(x, model_input[:2], method=tf.image.ResizeMethod.BILINEAR))(x)

        self.model = Model(inputs=encoder.input, outputs=x)
        print("Combined model built successfully.")

    def train(self, train_l, ground_truth_ab, epochs=10, batch_size=32, learning_rate=0.001, model_file=f'colorizer_decoder.keras', overwrite=True):

        if self.model is None:
            print("Tried to train before model was built")
            exit()

        # Setup for training decoder
        optimizer = Adam(learning_rate=learning_rate)
        loss_fn = MeanSquaredError()

        # Compile model for training
        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
        # Train model
        print(f"Training VGG16 Transfer Colorizer model for {epochs} epochs...")
        history = self.model.fit(
            train_l,
            ground_truth_ab,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        # Save the model
        if os.path.exists(model_file) and not overwrite:
            print(f"Warning: {model_file} already exists and will not be overwritten.")
        else:
            self.decoder.save(model_file)
            print(f"Model saved to {model_file}")

        self.is_trained = True
        return history

    def train_with_generator(self, datagenerator, epochs=10, batch_size=32, learning_rate=0.001, model_file=f'colorizer_decoder.keras', overwrite=True):

        if self.model is None:
            print("Tried to train before model was built")
            exit()
    
        # Setup for training decoder
        optimizer = Adam(learning_rate=learning_rate)
        loss_fn = MeanSquaredError()

        # Compile decoder model for training
        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
        # Train decoder model
        print(f"Training VGG16 Transfer Colorizer model for {epochs} epochs...")
        history = self.model.fit(
            datagenerator,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        # Save the model
        if os.path.exists(model_file) and not overwrite:
            print(f"Warning: {model_file} already exists and will not be overwritten.")
        else:
            self.model.save(model_file)
            print(f"Model saved to {model_file}")

        self.is_trained = True
        return history

    def load_decoder(self, model_file):
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file '{model_file}' not found.")

        self.model = tf.keras.models.load_model(model_file)
        self.is_trained = True
        print(f"Model loaded from {model_file}")


