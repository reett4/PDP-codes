from datetime import datetime

from ultralytics import YOLO
import cv2
import numpy as np

"""
Detection demo for the halfway show. Parameters for visuals are compatible with the huge screen 
on stage, and require adjusting based on the dimensions of the screen, on which the demo is shown.
"""

# PARAMETERS FOR VISUALS
# window params
screen_width, screen_height = 4500, 2525 # has to correspond to the screen
border_thickness_ratio = 0.01
border_color = (225, 229, 231)
window_name = "PDP TEAM SAAB"
# cropping
y_start = 350; h = 1500; x_start = 500; w = 1050
# blink
blink_counter = 0
blink_frequency = 5
blink_color = (0, 0, 255)
# font
font = cv2.FONT_HERSHEY_SIMPLEX
# live view txt
live_text = "TEAM SAAB live view"
live_font_scale = 3.5
live_font_color = (1, 1, 1)
live_font_thickness = 9
# location txt
atm_text = "@ DF The Stage"
atm_font_scale = 3
atm_font_color = (1, 1, 1)
atm_font_thickness = 9
# opaque rectangle
top_left = (540, 65)
width, height = 1285, 135
bottom_right = (top_left[0] + width, top_left[1] + height)
color = (225, 229, 231)
opacity = 0.5

# DETECTION & CAPTURING SPECIFICATIONS
model = YOLO("HWG_demo/best.pt")
confidence_threshold = 0.5
cap = cv2.VideoCapture(0)

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# DETECTION LOOP
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    print("Press 'q' to quit the real-time detection.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture an image.")
            break

        # handle colors correctly
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        else:
            frame = frame
        
        frame = frame[y_start:y_start+h, x_start:x_start+w]
        frame_height, frame_width = frame.shape[:2]
        
        results = model(frame, conf=confidence_threshold, verbose=False, ) # false doesn't print results to terminal
        frame = results[0].plot()

        top_bottom_border = int(screen_height * border_thickness_ratio / 2)
        left_right_border = int(screen_width * border_thickness_ratio / 2)

        bordered_frame = cv2.copyMakeBorder(
            frame,
            top=top_bottom_border,
            bottom=top_bottom_border,
            left=left_right_border,
            right=left_right_border,
            borderType=cv2.BORDER_CONSTANT,
            value=border_color,
        )
        aspect_ratio = frame_width / frame_height

        if screen_width / screen_height > aspect_ratio:
            new_height = screen_height
            new_width = int(aspect_ratio * screen_height)
        else:
            new_width = screen_width
            new_height = int(screen_width / aspect_ratio)

        resized_frame = cv2.resize(bordered_frame, (new_width, new_height))
        top_pad = (screen_height - new_height) // 2
        bottom_pad = screen_height - new_height - top_pad
        left_pad = (screen_width - new_width) // 2
        right_pad = screen_width - new_width - left_pad

        frame = cv2.copyMakeBorder(
            resized_frame,
            top_pad, bottom_pad, left_pad, right_pad,
            borderType=cv2.BORDER_CONSTANT,
            value=border_color
        )
        overlay = frame.copy()
        cv2.rectangle(overlay, top_left, bottom_right, color, -1)
        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

        if (blink_counter // blink_frequency) % 2 == 0:
            center_x, center_y = 605, 132
            radius = 40
            thickness = -1
            cv2.circle(frame, (center_x, center_y), radius, blink_color, thickness)
        blink_counter += 1

        text_size = cv2.getTextSize(live_text, font, live_font_scale, live_font_thickness)[0]
        text_x = 690
        text_y = text_size[1] + 90
        cv2.putText(frame, live_text, (text_x, text_y), font, live_font_scale, live_font_color, live_font_thickness)

        text_size = cv2.getTextSize(atm_text, font, atm_font_scale, atm_font_thickness)[0]
        text_x = 560
        text_y = screen_height - 90
        cv2.putText(frame, atm_text, (text_x, text_y), font, atm_font_scale, atm_font_color, atm_font_thickness)

        current_time = datetime.now().strftime("%H:%M:%S")
        time_text_size = cv2.getTextSize(current_time, font, atm_font_scale, atm_font_thickness)[0]
        time_x = (screen_width - time_text_size[0]) // 2
        time_y = screen_height - 90
        cv2.putText(frame, current_time, (time_x, time_y), font, atm_font_scale, atm_font_color, atm_font_thickness)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()