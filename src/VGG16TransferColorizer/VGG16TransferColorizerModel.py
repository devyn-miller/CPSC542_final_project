import os
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Concatenate, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from skimage.color import rgb2lab, lab2rgb, gray2rgb

from helper import *

class VGG16TransferColorizer:
    def __init__(self, image_shape=(720, 1080)):
        self.input_shape = image_shape
        self.encoder = None
        self.decoder = None
        self.is_trained = False

    def predict(self, grayscale_image):
        if self.encoder is None or self.decoder is None:
            print("Tried to predict before model was built")
            exit()

        # Convert 1 channel grayscale to RGB (repeat across 3 channels)
        input_l = gray2rgb(grayscale_image).reshape(((1,) + self.input_shape + (3,)))
        # Create classification latent space
        bottleneck = self.encoder.predict(input_l)
        # Predict ab color channels from the bottleneck latent space
        output_ab = self.decoder.predict(bottleneck)

        return output_ab

    def build_model(self):
        model_input = (self.input_shape + (3,))
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
        self.decoder = Sequential(
            [
                Conv2D(256, (3,3), activation='relu', padding='same', 
                    input_shape=encoder_output_shape),
                Conv2D(128, (3,3), activation='relu', padding='same'),
                UpSampling2D((2, 2)),
                Conv2D(64, (3,3), activation='relu', padding='same'),
                UpSampling2D((2, 2)),
                Conv2D(32, (3,3), activation='relu', padding='same'),
                UpSampling2D((2, 2)),
                Conv2D(16, (3,3), activation='relu', padding='same'),
                UpSampling2D((2, 2)),
                Conv2D(2, (3, 3), activation='tanh', padding='same'),
                UpSampling2D((2, 2)),
                Lambda(lambda x: tf.image.resize(x, self.input_shape, method=tf.image.ResizeMethod.BILINEAR))
            ],
            name='ColorizationDecoder'
        )
        print("Model built! Ready to train.")

    def train(self, train_dataset, val_dataset, epochs=10, batch_size=32, learning_rate=0.001, model_file='model.keras', overwrite=True):
        if self.encoder is None or self.decoder is None:
            print("Tried to train before model was built")
            exit()

        optimizer = Adam(learning_rate=learning_rate)
        loss_fn = MeanSquaredError()

        print(f"Training VGG16 Transfer Colorizer model for {epochs} epochs...")
        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
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

    def load_model(self, model_file):
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file '{model_file}' not found.")

        self.model = tf.keras.models.load_model(model_file)
        self.is_trained = True
        print(f"Model loaded from {model_file}")

def main():
    # Hyperparameters
    TRAIN_EPOCHS: int = 10
    LEARNING_RATE = 0.001
    IMAGE_SHAPE: tuple = (720, 1280)
    MODEL_FILE: str = 'vgg16_colorizer_model.keras'
    DATA_DIR = './src/videos/'
    
    '''
    Use Ponthea data processing/loading scripts to load and split the dataset into train and validation sets.

    train_dataset, val_dataset = load_and_split_dataset(DATA_DIR, IMAGE_SHAPE, batch_size=32)

    colorizer = VGG16TransferColorizer(image_shape=IMAGE_SHAPE)
    colorizer.build_model()
    colorizer.train(train_dataset, val_dataset, epochs=TRAIN_EPOCHS, batch_size=32, learning_rate=LEARNING_RATE, model_file=MODEL_FILE)
    '''
    
    # Load the data
    # full_dataset = tf.keras.utils.image_dataset_from_directory(
    #     DATA_DIR,
    #     label_mode=None,
    #     color_mode='grayscale',
    #     batch_size=32,
    #     image_size=(IMAGE_SHAPE + (3,))
    # )

    train_dataset = None
    val_dataset = None

    # Create an instance of the VGG16TransferColorizer
    colorizer = VGG16TransferColorizer(image_shape=IMAGE_SHAPE)
    colorizer.build_model
    # Train the model

    history = colorizer.train(train_dataset, val_dataset, epochs=TRAIN_EPOCHS, learning_rate=LEARNING_RATE, model_file=MODEL_FILE)

    # Load a pre-trained model
    # colorizer.load_model('path/to/pre-trained/model.h5')

    # Make predictions
    # Continuously predict and visualize new images
    fig, ax = plt.subplots(figsize=(10, 6))
    val_iterator = iter(val_dataset)

    while True:
        try:
            test_image = next(val_iterator)
            predicted_color = colorizer.predict(test_image)

            # Display the predicted color image
            ax.clear()
            ax.imshow(predicted_color)
            ax.set_title('Predicted Color Image')
            ax.axis('off')
            plt.pause(2)  # Pause for 2 seconds before updating the image

        except StopIteration:
            # Reset the iterator when the end of the dataset is reached
            val_iterator = iter(val_dataset)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    main()
