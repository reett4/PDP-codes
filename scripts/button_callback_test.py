import Jetson.GPIO as GPIO
import time


""" 
    Reads button presses from two buttons while also simultaneously sleeping.
"""


# GPIO pin definitions
button1_pin = 42  # board pin 18
button2_pin = 13  # board pin 33

# GPIO setup
GPIO.setmode(GPIO.BOARD)
GPIO.setup(button1_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(button2_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def button1_callback(channel):
    time.sleep(0.05)
    input_state = GPIO.input(button1_pin)
    if input_state == 1:
        print("button 1 (pin 42) pressed")
    else:
        print("button 1 (pin 42) released")

def button2_callback(channel):
    time.sleep(0.05)
    input_state = GPIO.input(button2_pin)
    if input_state == 1:
        print("button 1 (pin 13) pressed")
    else:
        print("button 1 (pin 13) released")

# attach event listeners
GPIO.add_event_detect(button1_pin, GPIO.FALLING, callback=button1_callback, bouncetime=100)
GPIO.add_event_detect(button2_pin, GPIO.BOTH, callback=button2_callback, bouncetime=100)

print("listening for button presses on pins 42 and 13...")
try:
    while True:
        time.sleep(1)
        print("sleeping...")
except KeyboardInterrupt:
    GPIO.cleanup()
    print("clean exit!")
