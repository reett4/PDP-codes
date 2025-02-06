import time
import os

import cv2
from playsound import playsound


def capture_images(interval, total_images, start_number, sound_file):
    """
    Captures images from the default camera and makes a sound at acquisition. Images are written
    as .png to the folder 'images' with naming based on running indices, and overwritten if they 
    already exist.

            Parameters:
                    interval (int): Time in seconds in between acquisition of subsequent images
                    total_images (int): Total number of images to capture
                    start_number (int): First running index for naming the captured images.
                    sound_file (str): Relative path to the sound file played at acquisition
    """
    sound_file_path = os.path.abspath(sound_file)

    if not os.path.exists(sound_file_path):
        print(f"Error: Sound file '{sound_file_path}' does not exist.")
        return
    
    output_dir = "images"
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    image_count = 0
    current_number = start_number

    while image_count < total_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        filename = os.path.join(output_dir, f"{current_number:06d}.png")
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        
        try:
            playsound(sound_file_path)
        except Exception as e:
            print(f"Error playing sound: {e}")
        
        image_count += 1
        current_number += 1
        time.sleep(interval)
    
    cap.release()
    cv2.destroyAllWindows()
    print("Completed capturing images.")