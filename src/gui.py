import tkinter as tk
from preprocessing import process_all_videos

def on_enter():
    frame_count = frame_count_var.get()
    duration = duration_var.get()
    resolution = resolution_var.get()
    # Convert inputs to appropriate types
    frame_count = int(frame_count) if frame_count else None
    duration = int(duration) if duration else None
    resolution = tuple(map(int, resolution.split('x'))) if resolution else (1280, 720)
    
    # Call your processing function here
    process_all_videos('/path/to/videos', '../data', frame_count, duration, resolution)
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
