import importlib
import augmentation as augmentation
importlib.reload(augmentation)
from augmentation import ImageAugmenter

from objects.stack import Stack
import cv2
import os

<<<<<<< HEAD
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
        frame_path = os.path.join(image_location, f"frame{count}.jpg")
        cv2.imwrite(frame_path, image_resized)     # save frame as JPEG file      
        
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

=======
# https://medium.com/@Ralabs/the-beginners-guide-for-video-processing-with-opencv-aa744ec04abb
# https://www.geeksforgeeks.org/python-process-images-of-a-video-using-opencv/ 
def process_video(video_file_location, image_location='../data', resolution=(1280, 720)):
    '''
    Takes in a video file location, converts the video to 
    frames and then places them into a folder. 
    (if unspecified then it places it in the data folder)

    :param video_file_location: path to the video file
    :type video_file_location: str
    :param image_location: location of the saved images, defaults to '../data'
    :type image_location: str, optional
    :param resolution: resolution of the video, defaults to (1280, 720)
    :type resolution: tuple, optional
    '''
    # Step 1: Create VideoCapture object to read video 
    cap = cv2.VideoCapture(video_file_location)
    
    idx = 0
    
    # Step 2: Loop until end of video 
    while (cap.isOpened()): 
        # create frame-by-frame
        ret, frame = cap.read()
        frame = cv2.resize(frame, resolution)
        
        # display resulting frame 
        cv2.imshow('Frame', frame)
        
        # convert BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # convert BGR -> grayscale 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        cv2.imwrite(os.path.join(image_location, "/bgr", f"bgr_{idx}"), frame)
        cv2.imwrite(os.path.join(image_location, "/rgb", f"rgb_{idx}"), rgb)
        cv2.imwrite(os.path.join(image_location, "/gray", f"gray_{idx}"), gray)
        
        idx += 1
        
        # exit if q is pressed 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break 
        
        
    cap.release()
    cap.destoryAllWindows()
    
    return

def process_videos(video_file_directory, image_location='../data', resolution=(1280, 720)):
    '''
    Takes in all video file locations in a directory, 
    converts the video to frames and then places them into a folder. 
    (if unspecified then it places it in the data folder)

    :param video_file_directory: path to the video file directory
    :type video_file_directory: str
    :param image_location: location of the saved images, defaults to '../data'
    :type image_location: str, optional
    :param resolution: resolution of the video, defaults to (1280, 720)
    :type resolution: tuple, optional
    '''
    
    for file in video_file_directory:  
        if file.endswith('.mp4'): 
            video_file_location = os.path.join(video_file_directory, file)
            process_video(video_file_location)
    
    return 
    

def train_test_validation_split(stack, image_location='../data', ):
    '''
    Creates train/test/validation generators and returns them.
    
    Each dataset should be build as following:
    data['bw_image'] = 1080x720x1
    data['colored_image'] = 1080x720x3

    :param stack: The stack to be updated 
    :type stack: Stack
    :param image_location: _description_, defaults to '../data'
    :type image_location: str, optional
    :return: Stack updated with the test/train/validation 
    :rtype: Stack
    '''
    
    train_dataset = ''
    test_dataset = ''
    val_dataset = ''
    
    stack.update_datasets(train_dataset, test_dataset, val_dataset)
    return stack

def augment_datasets(stack):
    '''
    This updates the datasets with augmented images, up to 
    yall what type of augmentation you want to use just make 
    sure you use the ImageAugmenter class.

    :param stack: The stack to be augmented
    :type stack: Stack
    :return: The augmented stack 
    :rtype: Stack
    '''
    augmenter = ImageAugmenter(IMG_WIDTH=1080, IMG_HEIGHT=720)
    
    train_dataset = augmenter.augment(stack.train_dataset)
    test_dataset = augmenter.augment(stack.test_dataset)
    val_dataset = augmenter.augment(stack.val_dataset)
    
    stack.update_datasets(train_dataset, test_dataset, val_dataset)
    return stack
>>>>>>> b44aecf7a7633ac0ba5f1c7cb1533f76fd91aad9
    
    
def create_tts_directories(train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    src_directory = './data/images'

<<<<<<< HEAD
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
    process_all_videos('./data/movies', './data/images')
    stack = train_test_validation_split(stack, BATCH_SIZE, RESOLUTION)
=======
def batch_datasets(stack, BATCH_SIZE):
    '''
    Batches the train, test and validation sets based on 
    the BATCH_SIZE. BATCH_SIZE is going to depend on your computer.
    '''
    
    train_dataset = stack.train_dataset.batch(BATCH_SIZE)
    test_dataset = stack.test_dataset.batch(BATCH_SIZE)
    val_dataset = stack.val_dataset.batch(BATCH_SIZE)
    
    stack.update_datasets(train_dataset, test_dataset, val_dataset)
    return stack

def preprocess(video_file_directory, BATCH_SIZE = 8):
    '''
    This is the method called by main.ipynb. 
    It also calls all the other functions and 
    returns the stack which will hold the finished datasets.

    :param video_file_directory: path to the video file directory
    :type video_file_directory: str
    :param BATCH_SIZE: the batch size, defaults to 8
    :type BATCH_SIZE: int, optional
    :return: the stack with updated datasets
    :rtype: Stack
    '''
    stack = Stack()
    process_videos(video_file_directory)
    stack = train_test_validation_split(stack)
    stack = augment_datasets(stack)
    stack = batch_datasets(stack, BATCH_SIZE)
>>>>>>> b44aecf7a7633ac0ba5f1c7cb1533f76fd91aad9
    
    return stack