import cv2
import os

def process_video_frames(data_path='data'):
    if not os.path.exists(data_path):
        print(f"Error: The directory {data_path} does not exist.")
        return []

    frames = []  
    files = os.listdir(data_path)
    video_files = [f for f in files if f.endswith(('.mp4', '.avi', '.mov'))]  # Add other video formats as needed

    for video_file in video_files:
        video_path = os.path.join(data_path, video_file)
        print(f"Processing {video_file}...")

        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_file}.")
            continue

        while True:
            ret, frame = cap.read()
            
            if not ret:
                print(f"Finished processing {video_file}. Exiting ...")
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frames.append((frame, gray_frame))

        cap.release()


    return frames