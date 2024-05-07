from src.preprocessing import *
from src.VGG16TransferColorizer.VGG16TransferColorizerModel import VGG16TransferColorizer
# from src.objects.data import Data
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

MODEL_FILE: str = 'vgg16_colorizer_model.keras'
DATA_DIR = './src/videos/'
BATCH_SIZE: int = 32


'''
DATA GENERATION
'''

# Define data iterator
def data_generator(batch_size, df):
    num_samples = len(df)
    while True:
        # Shuffle the DataFrame for each epoch
        df_shuffled = df.sample(frac=1).reset_index(drop=True)
        
        for i in range(0, num_samples, batch_size):
            batch_df = df_shuffled.iloc[i:i+batch_size]
            batch_gray_images = []
            batch_rgb_images = []
            
            for index, row in batch_df.iterrows():
                # Load gray image (train)
                gray_image = np.array(Image.open(row['gray']))
                # Load rgb image (val)
                rgb_image = np.array(Image.open(row['rgb']))
                
                batch_gray_images.append(gray_image)
                batch_rgb_images.append(rgb_image)
                
            yield np.array(batch_gray_images), np.array(batch_rgb_images)

def show_images_from_generator(generator):
    # Get a batch of images from the generator
    gray_images, rgb_images = next(generator)
    
    # Plot the first 9 images
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(gray_images[i], cmap='gray')
        plt.title('Gray')
        plt.axis('off')
        
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(rgb_images[i])
        plt.title('RGB')
        plt.axis('off')
        
    plt.show()

data_dir = 'vibrant'
image_paths = dataset(data_dir, image_location='data/')
# Establish generator
datagenerator = data_generator(BATCH_SIZE, image_paths)

# Visualize
show_images_from_generator(datagenerator)