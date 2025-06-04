import cv2
import numpy as np
import sys
import Jetson.GPIO as GPIO
import time
import threading
from ultralytics import YOLO


"""
    A script for running a four-camera detection with the ability to start or stop it using a 
    button, used for demonstrating our prototype during Demo Day (early April).
"""


sys.stdout = open('/dev/null', 'w')
model = YOLO('HWS_demo/best.pt').to('cuda')
sys.stdout = sys.__stdout__

GST_PIPELINE_SENSOR_0 = (
    "nvarguscamerasrc sensor-id=0 tnr-mode=0 wb-mode=1 ! video/x-raw(memory:NVMM), width=656, height=494, format=NV12, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
)
GST_PIPELINE_SENSOR_1 = (
    "nvarguscamerasrc sensor-id=1 tnr-mode=0 wb-mode=1 ! video/x-raw(memory:NVMM), width=656, height=494, format=NV12, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
)
GST_PIPELINE_SENSOR_2 = (
    "nvarguscamerasrc sensor-id=2 tnr-mode=0 wb-mode=1 ! video/x-raw(memory:NVMM), width=656, height=494, format=NV12, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
)
GST_PIPELINE_SENSOR_3 = (
    "nvarguscamerasrc sensor-id=3 tnr-mode=0 wb-mode=1 ! video/x-raw(memory:NVMM), width=656, height=494, format=NV12, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
)

but_pin = 42 
GPIO.setmode(GPIO.BOARD)
GPIO.setup(but_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP) 

running = False; camera_thread = None
confidence_threshold = 0.5


"""
    Runs all four cameras and detects mines from their stream using the given model.
"""
def run_camera():
    global running
    cap_0 = cv2.VideoCapture(GST_PIPELINE_SENSOR_0, cv2.CAP_GSTREAMER)
    cap_1 = cv2.VideoCapture(GST_PIPELINE_SENSOR_1, cv2.CAP_GSTREAMER)
    cap_2 = cv2.VideoCapture(GST_PIPELINE_SENSOR_2, cv2.CAP_GSTREAMER)
    cap_3 = cv2.VideoCapture(GST_PIPELINE_SENSOR_3, cv2.CAP_GSTREAMER)

    if not all([cap_0.isOpened(), cap_1.isOpened(), cap_2.isOpened(), cap_3.isOpened()]):
        print("Error: Could not open one or more cameras.")
        return

    while running:
        ret_0, frame_0 = cap_0.read(); frame_0 = cv2.flip(frame_0, -1)
        ret_1, frame_1 = cap_1.read()
        ret_2, frame_2 = cap_2.read()
        ret_3, frame_3 = cap_3.read(); frame_3 = cv2.flip(frame_3, -1)

        if not all([ret_0, ret_1, ret_2, ret_3]):
            break

        results_0 = model(frame_0, conf=confidence_threshold, verbose=False)
        frame_0 = results_0[0].plot()

        results_1 = model(frame_1, conf=confidence_threshold, verbose=False)
        frame_1 = results_1[0].plot()

        results_2 = model(frame_2, conf=confidence_threshold, verbose=False)
        frame_2 = results_2[0].plot()

        results_3 = model(frame_3, conf=confidence_threshold, verbose=False)
        frame_3 = results_3[0].plot()

        top_row = np.hstack((frame_0, frame_1))
        bottom_row = np.hstack((frame_2, frame_3))
        combined_frame = np.vstack((top_row, bottom_row))
        cv2.imshow("Team SAAB Proto Demo Day", combined_frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            running = False
            break

    cap_0.release()
    cap_1.release()
    cap_2.release()
    cap_3.release()
    cv2.destroyAllWindows()
    print("Camera stopped.")


"""
    Callback function for the button.
"""
def button_callback(channel):
    global running, camera_thread

    button_state = GPIO.input(but_pin)
    print(f"Button state: {button_state}") 

    if button_state == GPIO.HIGH:
        if not running:
            running = True
            camera_thread = threading.Thread(target=run_camera)
            camera_thread.start()

    elif button_state == GPIO.LOW:
        if running:
            running = False
            if camera_thread and camera_thread.is_alive():
                camera_thread.join()
            print("Camera thread joined.")

GPIO.add_event_detect(but_pin, GPIO.BOTH, callback=button_callback, bouncetime=200)


try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    GPIO.cleanup()
    print("Exiting...")
