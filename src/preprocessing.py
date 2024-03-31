from augmentation import ImageAugmenter
from stack import Stack
import cv2
import os

def process_video(video_file_location, image_location='../data'):
    '''Takes in a video file location, converts the video to a 
    bunch of images and then places them into a folder. 
    (if unspecified then it places it in the data folder)
    '''
    vidcap = cv2.VideoCapture(video_file_location)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(os.path.join(image_location, f"frame{count}.jpg"), image)     # save frame as JPEG file      
        success, image = vidcap.read()
        count += 1

def process_all_videos(directory, image_location='../data'):
    for filename in os.listdir(directory):
        if filename.endswith(".mp4"):
            video_file_location = os.path.join(directory, filename)
            process_video(video_file_location, image_location)

def color_to_bw(colored_image):
    '''Turns a colored image into a black and white image.'''
    bw_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2GRAY)
    return bw_image

def train_test_validation_split(stack, image_location='../data'):
    '''Creates train/test/validation datasets.'''
    # Example logic to split datasets
    # This is a placeholder. You need to replace it with your actual data splitting logic
    train_dataset = "your_train_dataset"
    test_dataset = "your_test_dataset"
    val_dataset = "your_val_dataset"
    stack.update_datasets(train_dataset, test_dataset, val_dataset)
    return stack

def augment_datasets(stack):
    '''Updates the datasets with augmented images.'''
    augmenter = ImageAugmenter(IMG_WIDTH=1080, IMG_HEIGHT=720)
    # Example augmentation logic. Replace with actual augmentation logic
    augmented_train_dataset = augmenter.augment(stack.train_dataset)
    augmented_test_dataset = augmenter.augment(stack.test_dataset)
    augmented_val_dataset = augmenter.augment(stack.val_dataset)
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
    process_all_videos('/Users/devynmiller/Downloads/movies-cpsc542', '../data')
    stack = train_test_validation_split(stack)
    stack = augment_datasets(stack)
    stack = batch_datasets(stack, BATCH_SIZE)
    
    return stack