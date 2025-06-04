import atexit
import os
import time
from datetime import datetime

import cv2
from threading import Event, Thread, Lock
from ultralytics import YOLO
from pymavlink import mavutil


""" 
    A script for running 4-camera detection during flight. Saves both all captured images and the 
    detections to separate folders with camera indices, timestamps and GPS data (latitude, 
    longitude and altitude) in their filenames. 
"""


GST_PIPELINE_SENSOR_0 = (
    "nvarguscamerasrc sensor-id=0 tnr-mode=0 wb-mode=0 ! video/x-raw(memory:NVMM), width=3280, height=2464, format=NV12, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
)
GST_PIPELINE_SENSOR_1 = (
    "nvarguscamerasrc sensor-id=1 tnr-mode=0 wb-mode=0 ! video/x-raw(memory:NVMM), width=3280, height=2464, format=NV12, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
)
GST_PIPELINE_SENSOR_2 = (
    "nvarguscamerasrc sensor-id=2 tnr-mode=0 wb-mode=0 ! video/x-raw(memory:NVMM), width=3280, height=2464, format=NV12, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
)
GST_PIPELINE_SENSOR_3 = (
    "nvarguscamerasrc sensor-id=3 tnr-mode=0 wb-mode=0 ! video/x-raw(memory:NVMM), width=3280, height=2464, format=NV12, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
)

BASE_SAVE_PATH = "/home/jetson/Documents/flying"
ALL_IMG_DIR = os.path.join(BASE_SAVE_PATH, "all_images")
DETECTED_IMG_DIR = os.path.join(BASE_SAVE_PATH, "detections")

MODEL = YOLO('gala_demo/final_model.pt').to('cuda')
TARGET_FPS = 1

master = mavutil.mavlink_connection('/dev/ttyTHS3', baud=115200, mavversion=2)
gps_data = {
    'lat': 0.0,
    'lon': 0.0,
    'alt': 0.0
}
gps_lock = Lock()

""" 
    Gets the current time.
"""
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S:%f")

""" 
    Reads the GPS information over MAVLink from Pixhawk.
"""
def gps_reader(stop_event):
    while not stop_event.is_set():
        msg = master.recv_match(type='GPS_RAW_INT', blocking=False)
        if msg:
            with gps_lock:
                gps_data['lat'] = msg.lat / 1e7
                gps_data['lon'] = msg.lon / 1e7
                gps_data['alt'] = msg.alt / 1000.0
        time.sleep(0.05) 

""" 
    Gets GPS data with a lock.
"""
def get_gps():
    with gps_lock:
        return gps_data['lat'], gps_data['lon'], gps_data['alt']

""" 
    Processes a camera stream.
"""
def process_camera_stream(sensor_id, pipeline, target_fps, stop_event):
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print(f"camera {sensor_id} could not be opened.")
        return

    # stabilize cameras at boot
    for _ in range(20):
        cap.read()
        time.sleep(0.25)

    frame_interval = 1.0 / target_fps

    while not stop_event.is_set():
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print(f"failed to grab frame from camera {sensor_id}")
            break

        lat, lon, alt = get_gps()
        filename_base = f"cam{sensor_id}_{get_timestamp()}_lat{lat:.7f}_lon{lon:.7f}_alt{alt:.2f}.jpg"
        filepath_all = os.path.join(ALL_IMG_DIR, filename_base)
        filepath_det = os.path.join(DETECTED_IMG_DIR, filename_base)

        cv2.imwrite(filepath_all, frame)

        results = MODEL(frame, verbose=False)[0]
        if len(results.boxes):
            annotated_frame = results.plot()
            cv2.imwrite(filepath_det, annotated_frame)

        elapsed = time.time() - start_time
        time_to_sleep = frame_interval - elapsed
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)

    cap.release()


if __name__ == "__main__":
    stop_event = Event()
    atexit.register(stop_event.set)
    try:
        os.makedirs(ALL_IMG_DIR, exist_ok=True)
        os.makedirs(DETECTED_IMG_DIR, exist_ok=True)

        threads = []

        # Start GPS reader thread
        gps_thread = Thread(target=gps_reader, args=(stop_event,))
        gps_thread.start()
        threads.append(gps_thread)

        pipelines = [
            (0, GST_PIPELINE_SENSOR_0),
            (1, GST_PIPELINE_SENSOR_1),
            (2, GST_PIPELINE_SENSOR_2),
            (3, GST_PIPELINE_SENSOR_3),
        ]

        for sensor_id, pipeline in pipelines:
            t = Thread(target=process_camera_stream, args=(sensor_id, pipeline, TARGET_FPS, stop_event))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

    except KeyboardInterrupt:
        print("stopping all cameras...")
        stop_event.set()
