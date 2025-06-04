from datetime import datetime
import json
import socket
from threading import Thread, Event
import time

import cv2 as cv
import Jetson.GPIO as GPIO
from ultralytics import YOLO

import screen


""" 
    Streams mine detection data from the Jetson over to the receiver computer.
"""


model = YOLO('gala_demo/final_model.pt').to('cuda')
HOST = '172.20.10.6'; PORT = 5050  # configuration-specific (works when a MacOS is the receiver)
DETECTION_BUTTON_PIN = 13; DATA_BUTTON_PIN = 42; DISPLAY_DURATION = 4
GST_PIPELINE_SENSOR = (
    "nvarguscamerasrc sensor-id=0 tnr-mode=0 wb-mode=1 ! video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
)

jetson_state = 'idle'
bar_event = Event(); bar_thread = None
mine_classes = model.names


""" 
    Gets the state of the Jetson.
"""
def get_jetson_state():
    global jetson_state
    return jetson_state

""" 
    Sets the state of the Jetson.
"""
def set_jetson_state(state):
    global jetson_state
    jetson_state = state

""" 
    Updates the acquired information about the state of the Jetson and the mine detection results.
"""
def update(strm_socket, capture, interval):
    detection_results = detect_mines(capture)
    send_data(detection_results, strm_socket)
    time.sleep(interval)

""" 
    Sends data over the socket.
"""
def send_data(detection_results, strm_socket):
    time_stamp = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
    data = {
        "time": time_stamp,
        "jetson_state": get_jetson_state(),
        "mines": detection_results,
    }
    message = json.dumps(data) + "\n"
    strm_socket.sendall(message.encode('utf-8'))

""" 
    Detects mines using the given model.
"""
def detect_mines(capture):
    # do not read camera if detection isn't enabled
    if get_jetson_state() == 'detecting':
        ret, frame = capture.read()
        if ret:
            height, width, _ = frame.shape
            results = model(frame, iou=0.025, conf=0.25)[0]

            detections = {}

            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                class_name = mine_classes.get(cls_id, f"class_{cls_id}")

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)

                if class_name not in detections:
                    detections[class_name] = []

                detections[class_name].append((x_center, y_center))

            return detections

        else:
            print('failed to capture a frame, check the camera!')
            return -1
    else: return 0

""" 
    A callback function for starting the detection with the detection button.
"""
def detection_callback(channel):
    global bar_thread
    time.sleep(0.05)
    if GPIO.input(DETECTION_BUTTON_PIN) == 1:
        if get_jetson_state() == 'uploading':
            print('uploading overriding detection button')
        elif not bar_event.is_set():
            bar_event.set()
            bar_thread = Thread(target=screen.draw_progress_bar,
                                 args=(DISPLAY_DURATION, "Detecting mines...", bar_event))
            bar_thread.start()
            print("model button pressed, started detection loop")
            set_jetson_state('detecting')
    else:
        bar_event.clear()
        if bar_thread and bar_thread.is_alive():
            bar_thread.join()
            print("model button released, detection loop stopped")
            set_jetson_state('idle')

""" 
    A callback function for starting to upload data with the button - this isn't actually
    implemented, so this just shows the progress bar.
"""
def data_callback(channel):
    time.sleep(0.05)
    if GPIO.input(DATA_BUTTON_PIN) == 1:
        if get_jetson_state() == "detecting":
            print('detection overriding upload button')
        else:
            set_jetson_state('uploading')
            print("data button pressed")
            Thread(target=screen.draw_progress_bar(DISPLAY_DURATION, "Uploading data to USB...")).start()
            set_jetson_state('idle')
    else:
        print("data button released")

""" 
    Initializes buttons and their callback functions.
"""
def initialize_buttons():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(DETECTION_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(DATA_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.add_event_detect(DETECTION_BUTTON_PIN, GPIO.BOTH, callback=detection_callback, bouncetime=100)
    GPIO.add_event_detect(DATA_BUTTON_PIN, GPIO.RISING, callback=data_callback, bouncetime=100)

""" 
    Initializes all hardware, i.e., buttons and the screen.
"""
def initialize_hardware():
    initialize_buttons()
    screen.display_image('media/saab_text_logo.png', invert=True, crop=(5, 5, 40, 50))
    print("hardware initialization ok!")


if __name__ == "__main__":
    initialize_hardware()
    # initialize camera separately to have its capture as an accessible object
    cap = cv.VideoCapture(GST_PIPELINE_SENSOR, cv.CAP_GSTREAMER)
    time.sleep(2)
    if not cap.isOpened():
        print("could not open the camera, exiting!")
        exit()

    # warm up the camera to make it work correctly
    for _ in range(10):
        cap.read()
        time.sleep(0.5)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream_socket:
        stream_socket.connect((HOST, PORT))
        print(f"connected to {HOST}: {PORT}")
        while True:
            update(stream_socket, cap, 0.3)
