import importlib
import augmentation as augmentation
importlib.reload(augmentation)
from augmentation import ImageAugmenter

from objects.stack import Stack
import cv2
import os

import cv2
import os
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def process_video(video_file_location, image_location='./data', frame_interval=5, resolution=(1280, 720)):
    '''Takes in a video file location, converts the video to images, and saves every nth frame (defined by frame_interval) to a folder.'''
    # Ensure the directory exists
    if not os.path.exists(image_location):
        os.makedirs(image_location)

    vidcap = cv2.VideoCapture(video_file_location)
    success, image = vidcap.read()
    count = 0
    saved_frame_count = 0  # To count how many frames have been saved

    while success:
        if count % frame_interval == 0:  # Check if the current frame number is divisible by frame_interval
            image_resized = cv2.resize(image, resolution)
            frame_path = os.path.join(image_location, f"frame{saved_frame_count}.jpg")
            cv2.imwrite(frame_path, image_resized)  # Save frame as JPEG file
            saved_frame_count += 1
        count += 1  # Increment the total frame count regardless of whether you saved it
        success, image = vidcap.read()

    vidcap.release()  # Release the video capture object


def process_all_videos(directory, image_location, resolution=(1280, 720)):
    frame_count = input("Enter the number of frames you want to extract (leave blank for all): ")
    #duration = input("Enter the duration of the video to process in seconds (leave blank for full video): ")

    frame_count = int(frame_count) if frame_count else None
    #duration = int(duration) if duration else None

    for filename in os.listdir(directory):
        if filename.endswith(".mp4"):
            video_file_location = os.path.join(directory, filename)
            process_video(video_file_location, image_location, frame_interval=frame_count, resolution=resolution)

    
    
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
    
    def organize_images(base_dir):
        # Create a subdirectory if it doesn't exist
        sub_dir = os.path.join(base_dir, 'images')
        os.makedirs(sub_dir, exist_ok=True)

        # Move all image files into the new subdirectory
        for file_name in os.listdir(base_dir):
            old_path = os.path.join(base_dir, file_name)
            if os.path.isfile(old_path) and file_name.endswith(('.jpg', '.jpeg', '.png')):
                new_path = os.path.join(sub_dir, file_name)
                shutil.move(old_path, new_path)

    # Organize images in train and validation directories
    organize_images('./data/train')
    organize_images('./data/validation')
    organize_images('./data/test')


     

def train_test_validation_split(stack, BATCH_SIZE, RESOLUTION, image_location='./data'):
    '''Creates train/test/validation datasets.'''
    IMG_WIDTH, IMG_HEIGHT = RESOLUTION
    
    current_directory = os.getcwd()
    print("Current directory:", current_directory)
    
    create_tts_directories()
    
    
    augmenter = ImageAugmenter(IMG_WIDTH=IMG_WIDTH, IMG_HEIGHT=IMG_HEIGHT)
    
    # Create augmented train data generator
    train_data_gen = augmenter.augment()

    # Note: Keep test and validation data generators without augmentation
    test_data_gen = ImageDataGenerator(rescale=1./255)
    val_data_gen = ImageDataGenerator(rescale=1./255)
    
    
    def testing_generator(directory):
        print("Testing generator for directory:", directory)
        data_gen = ImageDataGenerator(rescale=1./255)
        generator = data_gen.flow_from_directory(
            directory,
            target_size=(480, 360),  # Change as per your requirement
            batch_size=32,
            class_mode=None  # No labels for autoencoder
        )

        print("Found", generator.samples, "images in", generator.directory)
        if generator.samples > 0:
            batch = next(generator)
            print("Batch shape:", batch.shape)
        else:
            print("No data to generate from.")

    # Example usage
    #testing_generator('./data/train')
    #testing_generator('./data/validation')
    
    
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
    
    #batch = next(train_generator)
    #print("Type of output from generator:", type(batch))
    #images = next(train_generator)
    #print("Shape of images:", images.shape)


    stack.update_dimensions(IMG_WIDTH, IMG_HEIGHT)
    stack.update_datasets(train_generator, test_generator, val_generator)
    return stack

def preprocess(BATCH_SIZE = 8, RESOLUTION = (480, 360)):
    '''This is the method called by main.ipynb.  It also calls 
    all the other functions and returns the stack which will hold 
    the finished datasets.
    '''
    stack = Stack()
    process_all_videos('./data/movies', './data/images', RESOLUTION)
    stack = train_test_validation_split(stack, BATCH_SIZE, RESOLUTION)
    
    return stack
