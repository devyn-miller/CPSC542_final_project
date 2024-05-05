import cv2
import os
import pandas as pd 
from sklearn.model_selection import train_test_split

from pytube import YouTube, Playlist
import ssl
from pytube.exceptions import AgeRestrictedError

ssl._create_default_https_context = ssl._create_unverified_context

def download_video(video_url, resolution='360p'): 
    '''
    download a video from youtube

    :param video_url: the url of the video 
    :type video_url: str
    :param resolution: the resolution of the video, defaults to '360p'
    :type resolution: str, optional
    '''
    try:
        youtube = YouTube(video_url)
        youtube = youtube.streams.get_by_resolution(resolution)
        youtube.download()
        print("Download successful")
    except AgeRestrictedError as e:
        print(f"Age restriction error for video: {video_url}")
    except Exception as e:
        print(f"An error has occurred: {e}")

def download_playlist(playlist_url, resolution='360p'): 
    '''
    download all videos in a playlist

    :param playlist_url: the url of the playlist
    :type playlist_url: str
    :param resolution: the resolution of the video, defaults to '360p'
    :type resolution: str, optional
    '''
    playlist = Playlist(playlist_url)
    for video_url in playlist.video_urls: 
        download_video(video_url, resolution)

def process_video(video_file_location, file_num, image_location='../data', resolution=(1280, 360)):
    '''
    Takes in a video file location, converts the video to 
    frames and then places them into a folder. 
    (if unspecified then it places it in the data folder)

    :param video_file_location: path to the video file
    :type video_file_location: str
    :param image_location: location of the saved images, defaults to '../data'
    :type image_location: str, optional
    :param resolution: resolution of the video, defaults to (640, 360)
    :type resolution: tuple, optional
    '''
    # Step 1: Create VideoCapture object to read video 
    cap = cv2.VideoCapture(video_file_location)
    
    idx = 0
    frame_skip = 10  # Skip every 10 frames
    
    # Step 2: Loop until end of video 
    while (cap.isOpened()): 
        # Skip frames
        for _ in range(frame_skip):
            cap.grab()  # Skip frames without decoding
        
        # Read the frame
        ret, frame = cap.read()
        
        # Check if frame reading was successful
        if not ret:
            print("breaking")
            break
        
        # Check if the frame is empty
        if frame is None:
            print(f"Warning: Empty frame detected at index {idx}. Skipping...")
            continue
        
        # Process the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(image_location, "./rgb", f"{file_num}_rgb_{idx}.jpeg"), frame)
        cv2.imwrite(os.path.join(image_location, "./gray", f"{file_num}_gray_{idx}.jpeg"), gray)
        
        idx += 1
        
        # exit if q is pressed 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break 
        
    cap.release()
    print('video processed')
    return


def process_videos(video_file_directory, image_location='../data', resolution=(640, 360)):
    '''
    Takes in all video file locations in a directory, 
    converts the video to frames and then places them into a folder. 
    (if unspecified then it places it in the data folder)

    :param video_file_directory: path to the video file directory
    :type video_file_directory: str
    :param image_location: location of the saved images, defaults to '../data'
    :type image_location: str, optional
    :param resolution: resolution of the video, defaults to (640, 360)
    :type resolution: tuple, optional
    '''
    idx = 0; 
    for file in os.listdir(video_file_directory):  
        if file.endswith('.mp4'): 
            video_file_location = os.path.join(image_location, video_file_directory, file)
            process_video(video_file_location, idx)
            idx += 1
    
    return 
    
def dataset(image_location='../data'):
    types = []
    rgbs = []
    grays = []

    # Iterate through each subdirectory
    for subdir in os.listdir(image_location):
        subdir_path = os.path.join(image_location, subdir)
        if os.path.isdir(subdir_path):
            # Get the type (old, cartoon, vibrant, dark) from the subdirectory name
            type_name = subdir
            
            # Get the paths to rgb and gray directories
            rgb_path = os.path.join(subdir_path, 'rgb')
            gray_path = os.path.join(subdir_path, 'gray')
            
            # Get the list of files in rgb and gray directories
            rgb_files = os.listdir(rgb_path)
            gray_files = os.listdir(gray_path)
            
            # Assuming the files are corresponding pairs (same name in both directories)
            for rgb_file, gray_file in zip(rgb_files, gray_files):
                # Append the type for each file
                types.append(type_name)
                
                # Append the full paths of the rgb and gray files
                rgbs.append(os.path.join(rgb_path, rgb_file))
                grays.append(os.path.join(gray_path, gray_file))

    # Create DataFrame
    data = {'type': types, 'rgb': rgbs, 'gray': grays}
    df = pd.DataFrame(data)
    
    return df 
    
def normalize(X_train, y_train, X_test, y_test, X_val, y_val): 
    X_train = X_train.astype('float32') / 255 
    y_train = y_train.astype('float32') / 255
    
    X_test = X_test.astype('float32') / 255
    y_test = y_test.astype('float32') / 255
    
    X_val = X_val.astype('float32') /255 
    y_val = y_val.astype('float32') / 255   
    
    return X_train, y_train, X_test, y_test, X_val, y_val

def train_test_validation_split(image_location='../data'):
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
    
    df = dataset(image_location)
    
    X = df['gray']
    y = df['rgb']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=None)
    
    X_train, y_train, X_test, y_test, X_val, y_val = normalize(X_train, y_train, X_test, y_test, X_val, y_val)
    
    return X_train, y_train, X_test, y_test, X_val, y_val
    
    
