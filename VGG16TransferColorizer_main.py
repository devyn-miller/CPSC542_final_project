from src.VGG16TransferColorizer.VGG16TransferColorizerModel import VGG16TransferColorizer
from src.preprocessing import *
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb, gray2rgb

'''
HYPERPARAMETERS
'''

TRAIN_EPOCHS: int = 10
LEARNING_RATE = 0.001
IMAGE_SHAPE: tuple = (360,640)
BATCH_SIZE: int = 32
MODEL_FILE: str = 'vgg16_colorizer_model.keras'
DATA_DIR = './src/videos/'

print("working")

# Define data iterator 
# Helps to decrease memory usage
def data_generator(batch_size, df):
    num_samples = len(df)
    indices = np.arange(num_samples)

    while True:
        # Shuffle the indices for each epoch
        np.random.shuffle(indices)
        
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_df = df.iloc[batch_indices]
            batch_gray_images = []
            batch_ab_channels = []
            
            for index, row in batch_df.iterrows():
                # Load gray image (train) and resize
                gray_image = np.array(Image.open(row['gray']))
                resized_gray_image = np.array(Image.fromarray(gray_image).resize((IMAGE_SHAPE[1], IMAGE_SHAPE[0])))
                # Convert gray image to RGB format
                resized_gray_image_rgb = np.stack((resized_gray_image,) * 3, axis=-1)

                # Load rgb image (val) and resize
                rgb_image = np.array(Image.open(row['rgb']))
                resized_rgb_image = np.array(Image.fromarray(rgb_image).resize((IMAGE_SHAPE[1], IMAGE_SHAPE[0])))
                # Convert RGB image to LAB color space
                lab_image = rgb2lab(resized_rgb_image)
                # Extract only the AB channels
                ab_channels = lab_image[:, :, 1:]

                batch_gray_images.append(resized_gray_image_rgb)
                batch_ab_channels.append(ab_channels)
                
            yield np.array(batch_gray_images), np.array(batch_ab_channels)

def combine_grey_ab(gray_image, ab_channels):
    # Convert grayscale image to LAB color space
    lab_image = rgb2lab(gray_image)
    
    # Replace the AB channels in the LAB image with the provided AB channels
    lab_image[:, :, 1:] = ab_channels
    
    # Convert the LAB image back to RGB for return
    return lab2rgb(lab_image)

# display generator grey images, and combined with color channels
# -> for testing & validation purposes
def display_images_from_generator(generator):
    # Get a batch of images from the generator
    gray_images, ab_channels = next(generator)
    print(gray_images.shape)
    print(ab_channels.shape)
    
    # Plot the first 9 images
    plt.figure(figsize=(15, 10))
    for i in range(5):
        
        combined_image =  combine_grey_ab(gray_images[i], ab_channels[i])
        

        # Display the grayscale image
        plt.subplot(2, 5, i + 1)
        plt.imshow(gray_images[i], cmap='gray')
        plt.title('Grayscale')
        plt.axis('off')
        
        # Display the combined image
        plt.subplot(2, 5, i + 6)
        plt.imshow(combined_image)
        plt.title('Combined')
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

def main():

    '''
    DATA GENERATION
    '''
    # Using Ponthea dataset filepath generator
    data_dir = 'vibrant'
    image_paths_df = dataset(data_dir, image_location='data/')
    steps_per_epoch = len(image_paths_df) // BATCH_SIZE
    # Establish generator
    datagenerator = data_generator(BATCH_SIZE, image_paths_df)

    # Display plots of gray/color counterparts in training data
    display_images_from_generator(datagenerator)
    # return
    '''
    MODEL TRAINING
    '''

    # Create an instance of the VGG16TransferColorizer
    colorizer = VGG16TransferColorizer(image_shape=IMAGE_SHAPE)
    # Build model
    colorizer.build_model()
    colorizer.model.summary()
    # Train model
    history = colorizer.train_with_generator(
        datagenerator,
        steps_per_epoch=steps_per_epoch,
        epochs=TRAIN_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        model_file=MODEL_FILE,
    )

    loss = history.history['loss']  # Extract loss values
    accuracy = history.history['accuracy']  # Extract accuracy values
    print(f"Training loss: {loss}")
    print(f"Training accuracy: {accuracy}")

if __name__ == '__main__':
    main()