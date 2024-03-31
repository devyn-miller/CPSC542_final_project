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
    '''Turns a colored image into a black and white image.  Called 
    by train_test_validation_split to create the data['bw_image'] images.
    '''
    return bw_image

def train_test_validation_split(stack, image_location='../data', ):
    '''Creates train/test/validation generators and returns them.
    Each dataset should be build as following:
    data['bw_image'] = 1080x720x1
    data['colored_image'] = 1080x720x3
    We can change the sizes if they are too big.
    '''
    stack.update_datasets(train_dataset, test_dataset, val_dataset)
    return stack

def augment_datasets(stack):
    '''This updates the datasets with augmented images, up to 
    yall what type of augmentation you want to use just make 
    sure you use the ImageAugmenter class.
    '''
    augmenter = ImageAugmenter(IMG_WIDTH=1080, IMG_HEIGHT=720)
    stack.update_datasets(train_dataset, test_dataset, val_dataset)
    return stack
    

def batch_datasets(stack, BATCH_SIZE):
    '''Batches the train, test and validation sets based on 
    the BATCH_SIZE.  BATCH_SIZE is going to depend on your computer.
    '''
    stack.update_datasets(train_dataset, test_dataset, val_dataset)
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