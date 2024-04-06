from augmentation import ImageAugmenter
from objects.stack import Stack
import cv2
import os

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
    
    return stack