All the code cooked during [PDP](https://pdp.fi) (iteration 2024-2025, team Saab), organized clearly :star::microscope::grey_exclamation: Generally, brief descriptions of the given file are given after imports with functions also described shortly.

## Contents

The structure of the repository is as follows:

* **HWS_demo** contains the demo for the Halfway Show
    * `hws_demo.py` runs the live demo used in the presentation
    * `best.pt` is the model trained for the demo
* **gala_demo** contains the demo for the Gala
    * **assets** contains images used for the interface on the receiver side and `transformation_matrix.txt` used for a coordinate transformation 
    * `final_model.pt` is the final model trained by us for detecting PFM-1s and 9N235s
    * `jetson_socket.py` creates the interface that displays the state and detection information read from Jetson over the socket
    * `screen.py` implements the functionalities for the IC2 screen on the sensor module
    * `stream_demo.py` sends the detection and state information from Jetson over the socket to the receiver 
* **scripts** contains general scripts used for various tasks for the project
    * **utils** contains miscellaneous files useful for the scripts
    * `button_callback_test.py` reads button presses from two buttons
    * `capture_imgs.py` is a function that can be used to capture test data
    * `datacollection_dual.py` captures images from two cameras on a Raspberry Pi
    * `four_camera_button.py` is the demo used in the Demo Day
    * `test_detect.py` tests detection with a given model from one camera
    * `test_usb.py` implements the logic for uploading data from Jetson to USB
* `fly_detect_new.py` is to be run from boot to detect mines when flying the drone, and saves both detected mines and all acquired frames

## Notes

* All demonstration interfaces (i.e., the windows displayed and updated by the code) are scaled to the screens used in the demonstration. As these have been big screens, the windows don't properly fit laptop screens.
* All code has been tested to function on Jetson (with our specific configuration) or the demonstration platform (a laptop), and isn't therefore guaranteed to work in any environment.