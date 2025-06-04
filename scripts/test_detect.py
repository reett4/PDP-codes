import cv2
from ultralytics import YOLO


""" 
    A test script for testing the detection with one camera and the given model.
"""


model = YOLO('gala_demo/final_model.pt').to('cuda')
GST_PIPELINE_SENSOR = (
    "nvarguscamerasrc sensor-id=0 tnr-mode=0 wb-mode=1 ! video/x-raw(memory:NVMM), width=656, height=494, format=NV12, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
)

cap = cv2.VideoCapture(GST_PIPELINE_SENSOR, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

window_scale = 0.5 

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    results = model(frame, iou=0.025, conf=0.3)[0]
    for result in results:
        for box in result.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = map(int, box[:6])
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    frame_resized = cv2.resize(frame, (int(frame.shape[1] * window_scale), int(frame.shape[0] * window_scale)),
        interpolation=cv2.INTER_AREA)
    cv2.imshow("YOLOv8 Detection - Camera", frame_resized)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
