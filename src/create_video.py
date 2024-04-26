import cv2
import os

def process_video_frames(data_path='data', frame_skip=30) -> list[tuple]:
    if not os.path.exists(data_path):
        print(f"Error: The directory {data_path} does not exist.")
        return []

    frames = []  
    files = os.listdir(data_path)
    video_files = [f for f in files if f.endswith(('.mp4', '.avi', '.mov'))]  # Add other video formats as needed

    for video_file in video_files:
        video_path = os.path.join(data_path, video_file)
        print(f"Processing {video_file}... Storing every {frame_skip} frames")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video {video_file}.")
        else:
            current_frame = 0
            frames = []  # Ensure this list is defined to store your frames

            while True:
                ret, frame = cap.read()

                if not ret:
                    print(f"{current_frame} total frames")
                    print(f"Finished processing {video_file}. Exiting ...")
                    break

                # Process only every Nth frame (N=frame_skip)
                if current_frame % frame_skip == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frames.append((rgb_frame, gray_frame))

                current_frame += 1

            cap.release()


    return frames