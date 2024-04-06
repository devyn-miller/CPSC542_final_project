from augmentation import ImageAugmenter
from objects.stack import Stack
import cv2
import os

# https://medium.com/@Ralabs/the-beginners-guide-for-video-processing-with-opencv-aa744ec04abb
# https://www.geeksforgeeks.org/python-process-images-of-a-video-using-opencv/ 
def process_video(video_file_location, image_location='../data', resolution=(1280, 720)):
    '''
    Takes in a video file location, converts the video to a 
    bunch of images and then places them into a folder. 
    (if unspecified then it places it in the data folder)
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
    stack = train_test_validation_split(stack)
    stack = augment_datasets(stack)
    stack = batch_datasets(stack, BATCH_SIZE)
    
    return stack