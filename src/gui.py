import tkinter as tk
from src.preprocessing import process_all_videos

def on_enter():
    frame_count = frame_count_var.get()
    duration = duration_var.get()
    resolution = resolution_var.get()
    # Convert inputs to appropriate types
    frame_count = int(frame_count) if frame_count else None
    duration = int(duration) if duration else None
    resolution = tuple(map(int, resolution.split('x'))) if resolution else (1280, 720)
    
    # Adjust the call to match the expected signature of process_all_videos
    process_all_videos({
        'video_dir': '../videos',
        'data_dir': '../data',
        'frame_count': frame_count,
        'duration': duration,
        'resolution': resolution
    })
    root.destroy()

root = tk.Tk()
root.title("Dataset Creation")

frame_count_var = tk.StringVar()
duration_var = tk.StringVar()
resolution_var = tk.StringVar()

tk.Label(root, text="Number of frames:").pack()
tk.Entry(root, textvariable=frame_count_var).pack()

tk.Label(root, text="Duration (seconds):").pack()
tk.Entry(root, textvariable=duration_var).pack()

tk.Label(root, text="Resolution (width x height):").pack()
tk.Entry(root, textvariable=resolution_var).pack()

tk.Button(root, text="Enter", command=on_enter).pack()

root.mainloop()
