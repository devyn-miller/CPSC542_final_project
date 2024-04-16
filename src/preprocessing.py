import importlib
import augmentation as augmentation
importlib.reload(augmentation)
from augmentation import ImageAugmenter

from objects.stack import Stack

import cv2
import os
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def process_video(video_file_location, image_location='./data', frame_count=None, duration=None, resolution=(1280, 720)):
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

def process_all_videos(directory, image_location='./data'):
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

    
    
def create_tts_directories(train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    src_directory = './data/images'

    # Desired split
    train_ratio = 0.7
    val_ratio = 0.1
    test_ratio = 0.2

    current_directory = os.getcwd()
    print("Current directory2:", current_directory)

    # Create directories for the split if they don't exist
    if not os.path.exists('./data/train'):
        os.makedirs('./data/train')
    if not os.path.exists('./data/validation'):
        os.makedirs('./data/validation')
    if not os.path.exists('./data/test'):
        os.makedirs('./data/test')

    # Get all filenames in the source directory
    all_filenames = os.listdir(src_directory)
    all_filenames = [f for f in all_filenames if os.path.isfile(os.path.join(src_directory, f))]

    # Shuffle array to mix up images
    np.random.shuffle(all_filenames)

    # Calculate split points
    total_images = len(all_filenames)
    train_end = int(train_ratio * total_images)
    val_end = train_end + int(val_ratio * total_images)

    # Split filenames
    train_filenames = all_filenames[:train_end]
    val_filenames = all_filenames[train_end:val_end]
    test_filenames = all_filenames[val_end:]

    # Function to copy files to new directories
    def copy_files(filenames, dst_directory):
        for filename in filenames:
            src_path = os.path.join(src_directory, filename)
            dst_path = os.path.join(dst_directory, filename)
            shutil.copy(src_path, dst_path)

    # Copy files to the respective directories
    copy_files(train_filenames, './data/train')
    copy_files(val_filenames, './data/validation')
    copy_files(test_filenames, './data/test')
     

def train_test_validation_split(stack, BATCH_SIZE, image_location='./data'):
    '''Creates train/test/validation datasets.'''
    IMG_WIDTH=1080
    IMG_HEIGHT=720
    
    current_directory = os.getcwd()
    print("Current directory:", current_directory)
    
    create_tts_directories()
    
    
    augmenter = ImageAugmenter(IMG_WIDTH=1080, IMG_HEIGHT=720)
    
    # Create augmented train data generator
    train_data_gen = augmenter.augment()

    # Note: Keep test and validation data generators without augmentation
    test_data_gen = ImageDataGenerator(rescale=1./255)
    val_data_gen = ImageDataGenerator(rescale=1./255)
    
    current_directory = os.getcwd()
    print("Current directory3:", current_directory)
    
    train_generator = train_data_gen.flow_from_directory(
        './data/train',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode=None)

    test_generator = test_data_gen.flow_from_directory(
        './data/test',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode=None)

    val_generator = val_data_gen.flow_from_directory(
        './data/validation',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode=None)
    
    stack.update_datasets(train_generator, test_generator, val_generator)
    return stack

def augment_datasets(stack):
    '''Updates the datasets with augmented images.'''
    
    
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
    # Example batching logic. Replace with actual batching logic
    batched_train_dataset = stack.train_dataset.batch(BATCH_SIZE)
    batched_test_dataset = stack.test_dataset.batch(BATCH_SIZE)
    batched_val_dataset = stack.val_dataset.batch(BATCH_SIZE)
    stack.update_datasets(batched_train_dataset, batched_test_dataset, batched_val_dataset)
    return stack
def preprocess(BATCH_SIZE = 8):
    '''This is the method called by main.ipynb.  It also calls 
    all the other functions and returns the stack which will hold 
    the finished datasets.
    '''
    stack = Stack()
    process_all_videos('./data/movies', './data/images')
    stack = train_test_validation_split(stack, BATCH_SIZE)
    
    return stack