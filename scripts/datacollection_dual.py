import time
import os
import sys
from datetime import datetime

from picamera2 import Picamera2
from PIL import Image

"""
    Data gathering script to be run on Raspberry Pi with two cameras.
"""

# initialize and configure the two cameras
picam2a = Picamera2(0)
picam2b = Picamera2(1)
target_resolution = (2560, 1440)

config_a = picam2a.create_still_configuration(main={"size": target_resolution})
config_b = picam2b.create_still_configuration(main={"size": target_resolution})

picam2a.configure(config_a)
picam2b.configure(config_b)

# choose the directory to save images, create the directory if it doesn't exist
test_mode = "test" in sys.argv
print(test_mode)
save_directory = "/home/pdp/Desktop/Test" if test_mode else "/home/pdp/Desktop/Uusi"
os.makedirs(save_directory, exist_ok=True)

picam2a.start(); picam2b.start()

# function to capture an image, save it, and rotate it 180 degrees
def capture_image(camera, filename):
    camera.capture_file(filename)

    img = Image.open(filename)
    img = img.rotate(180)
    img.save(filename)
    print(f"image saved as {filename}")

print("press 'f' or 'p' to take a picture. Press 'q' to quit.")

counter = 0 

# loop to keep capturing images until 'q' is pressed
while True:
    user_input = input()
    if user_input == 'f' or user_input == 'p':
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_a = os.path.join(save_directory, f"camera_a_{timestamp}.jpg")
        filename_b = os.path.join(save_directory, f"camera_b_{timestamp}.jpg")

        print(f"key '{user_input}' pressed! Taking pictures...")
 
        capture_image(picam2a, filename_a)
        counter += 1 
        capture_image(picam2b, filename_b)
        counter += 1 

        print("taken " + str(counter) + " pictures so far!")

    elif user_input == 'q':
        print("quitting...")
        break
