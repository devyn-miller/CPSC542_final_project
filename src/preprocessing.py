try:
    from objects.augmentation import ImageAugmenter
    from objects.stack import Stack
except ImportError as e:
    print(e, "\nMake sure 'objects' module is installed and accessible")
import cv2
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

def process_video(video_file_location, image_location='../data', frame_count=None, duration=None, resolution=(1280, 720)):
    '''Takes in a video file location, converts the video to a 
    bunch of images and then places them into a folder. 
    (if unspecified then it places it in the data folder)
    '''
    vidcap = cv2.VideoCapture(video_file_location)
    success, image = vidcap.read()
    count = 0
    while success:
        if frame_count and count >= frame_count:
            break
        image_resized = cv2.resize(image, resolution)
        cv2.imwrite(os.path.join(image_location, f"frame{count}.jpg"), image_resized)     # save frame as JPEG file      
        success, image = vidcap.read()
        count += 1

def process_all_videos(directory, image_location='../data'):
    frame_count = input("Enter the number of frames you want to extract (leave blank for all): ")
    duration = input("Enter the duration of the video to process in seconds (leave blank for full video): ")
    resolution_input = input("Enter the desired resolution as width x height (leave blank for 720p): ")

    frame_count = int(frame_count) if frame_count else None
    duration = int(duration) if duration else None
    resolution = tuple(map(int, resolution_input.split('x'))) if resolution_input else (1280, 720)

    for filename in os.listdir(directory):
        if filename.endswith(".mp4"):
            video_file_location = os.path.join(directory, filename)
            process_video(video_file_location, image_location, frame_count=frame_count, duration=duration, resolution=resolution)

def color_to_bw(colored_image):
    '''Turns a colored image into a black and white image.'''
    bw_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2GRAY)
    return bw_image

def train_test_validation_split(stack, image_location='../data'):
    # Assuming a function to load your dataset exists within this file or is imported correctly
    dataset = stack.load_dataset(image_location)  # Corrected to use a method from the stack object to load the dataset
    
    # Split dataset into training and temp (test + validation) datasets
    train_dataset, temp_dataset = train_test_split(dataset, test_size=0.4, random_state=42)
    # Split temp dataset into testing and validation datasets
    test_dataset, val_dataset = train_test_split(temp_dataset, test_size=0.5, random_state=42)
    
    # Update stack with the new datasets
    stack.update_datasets(train_dataset, test_dataset, val_dataset)
    return stack

def augment_datasets(stack):
    '''Updates the datasets with augmented images.'''
    augmenter = ImageAugmenter(IMG_WIDTH=1080, IMG_HEIGHT=720)
    
    # Define a function that will be applied to each element of the dataset
    def augment_image(image):
        # Assuming the dataset consists of image-label pairs
        augmented_image = augmenter.augment(image[0])
        return (augmented_image, image[1])
    
    # Apply the augment_image function to each element of the dataset using map
    augmented_train_dataset = stack.train_dataset.map(augment_image)
    augmented_test_dataset = stack.test_dataset.map(augment_image)
    augmented_val_dataset = stack.val_dataset.map(augment_image)
    
    stack.update_datasets(augmented_train_dataset, augmented_test_dataset, augmented_val_dataset)
    return stack
    

def batch_datasets(stack, BATCH_SIZE):
    '''Batches the train, test, and validation sets.'''

    # Convert datasets to TensorFlow datasets and apply batching
    batched_train_dataset = tf.data.Dataset.from_tensor_slices(stack.train_dataset).batch(BATCH_SIZE)
    batched_test_dataset = tf.data.Dataset.from_tensor_slices(stack.test_dataset).batch(BATCH_SIZE)
    batched_val_dataset = tf.data.Dataset.from_tensor_slices(stack.val_dataset).batch(BATCH_SIZE)
    
    stack.update_datasets(batched_train_dataset, batched_test_dataset, batched_val_dataset)
    return stack

def preprocess(BATCH_SIZE = 8):
    '''This is the method called by main.ipynb.  It also calls 
    all the other functions and returns the stack which will hold 
    the finished datasets.
    '''
    stack = Stack()
    process_all_videos('/Users/devynmiller/CPSC542_final_project/data/movie_mp4s')
    stack = train_test_validation_split(stack)
    stack = augment_datasets(stack)
    stack = batch_datasets(stack, BATCH_SIZE)
    
    return stack


