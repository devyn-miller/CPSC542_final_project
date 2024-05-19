import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import cv2

class ImageDataGenerator(Sequence):
    def __init__(self, directory, batch_size=32, is_train = True, resolution=(480, 360)):
        """
        Initialize the data generator.
        :param directory: Path to the directory containing images.
        :param batch_size: Number of images per batch.
        """
        self.directory = directory  # Store the path to the directory where images are stored.
        self.batch_size = batch_size  # Store the batch size.
        self.is_train = is_train  # Indicates if the generator is used for training or validation.
        self.resolution = resolution  # Resize resolution of the images.

        # List all jpeg files in the directory and store their paths.
        self.image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.jpeg')]
        self.on_epoch_end()  # Initial shuffling of image files.

        self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,       # Random rotations from 0 to 10 degrees
            # width_shift_range=0.2,   # Random horizontal shifts
            # height_shift_range=0.2,  # Random vertical shifts
            # shear_range=1,         # Shear transformations
            zoom_range=.3,          # Random zoom
            horizontal_flip=True,    # Enable horizontal flipping
            fill_mode='nearest'      # Strategy for filling in newly created pixels
        )


    def __len__(self):
        """
        Return the number of batches per epoch.
        """
        return int(np.floor(len(self.image_files) / self.batch_size))  # Compute the number of batches.

    def __getitem__(self, index):
        """
        Generate one batch of data (input and output).
        :param index: Index of the batch in the sequence.
        :return: Tuple (input_batch, output_batch).
        """
        start_index = index * self.batch_size  # Calculate start index of the batch.
        end_index = (index + 1) * self.batch_size  # Calculate end index of the batch.
        batch_files = self.image_files[start_index:end_index]  # Slice the file list to get a batch.
        return self.__data_generation(batch_files)  # Generate data for the batch.

    def on_epoch_end(self):
        """
        Actions to be taken at the end of each epoch.
        """
        print("epoch ended")
        if self.is_train:  # Only shuffle if this is a training generator.
            np.random.shuffle(self.image_files)  # Shuffle the order of the input images.

    def __data_generation(self, batch_files):
        """
        Produces data containing batch_size samples - X : (n_samples, *dim, n_channels) and Y : (n_samples, *dim, 2).
        :param batch_files: List of image files to process for the batch.
        :return: A batch of processed images (input_batch, output_batch).
        """
        input_batch = []  # Initialize an empty list to store the processed input images (grayscale RGB).
        output_batch = []  # Initialize an empty list to store the processed output images (AB channels).
        for file in batch_files:
            img = cv2.imread(file)  # Read the image file.

            if self.is_train:
                # Apply data augmentation during training
                img = self.datagen.random_transform(img)

            img = cv2.resize(img, self.resolution)  # Resize the image to ensure consistency.

            # Prepare input image (grayscale RGB)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale.
            img_gray_rgb = np.stack((img_gray,)*3, axis=-1)  # Stack grayscale image across channel dimension to mimic RGB.
            input_batch.append(img_gray_rgb)

            # Prepare output image (AB channels)
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # Convert to LAB color space.
            img_ab = img_lab[:, :, 1:]  # Use only AB channels.
            output_batch.append(img_ab)
        
        return np.array(input_batch, dtype=np.uint8)/255, (np.array(output_batch, dtype=np.uint8) -127.5)/127.5  # Return the batches of processed images as numpy arrays.

