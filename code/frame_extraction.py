import cv2
import os

video_path = "/Users/binadisilva/Desktop/RUSL/IP/Project_IP/Videos/crack.MOV"
output_folder = "/Users/binadisilva/Desktop/RUSL/IP/Project_IP/Frames/crack"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file")
    exit()

frame_index = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Save every 10th frame
    if frame_index % 10 == 0:
        filename = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1
    
    frame_index += 1

cap.release()
print(f"Done! Saved {saved_count} frames.")