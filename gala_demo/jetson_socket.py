import sys
import socket
from threading import Thread
import queue
import json

import cv2 as cv
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QMainWindow, QHBoxLayout
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap


""" 
    Reads and prints data received over from a socket.

    Physical dimensions of the relevant objects:
    * PFM-1: width 12 cm
    * 9N235: width 24 cm
    * sandbox: width 119,5 cm, height 109,5 cm
"""


HOST = '0.0.0.0'; PORT = 5050   # configuration-specific (works on MacOS)
SCALING_RATIO = 10


def listen_socket(data_queue):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"listening on {HOST}:{PORT}...")
        while True:
            connection, address = s.accept()
            print(f"connection from {address}")
            data_queue.put({"__connection__": True})
            with connection:
                buffer = ""
                try:
                    while True:
                        data_chunk = connection.recv(1024).decode('utf-8')
                        if not data_chunk:
                            break
                        buffer += data_chunk
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            try:
                                data = json.loads(line)
                                data_queue.put(data)
                            except json.JSONDecodeError:
                                print("invalid JSON:", line)
                finally:
                    data_queue.put({"__connection__": False})


""" 
    Resizes an image to a target width.
"""
def resize_image(img, target_width):
    original_h, original_w = img.shape[:2]
    ratio = target_width / original_w
    target_height = (original_h * ratio)
    resized_img = cv.resize(img, (int(target_width), int(target_height)), interpolation=cv.INTER_AREA)
    return resized_img


""" 
    Class for the main window of the demo, inherits from QMainWindow.
"""
class Hiekkalaatikko(QMainWindow):
    def __init__(self):
        super().__init__()
        self.mines = []

        # -------- layout setup starts here --------
        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)
        central_widget.setStyleSheet("""
            #centralWidget {
                background-image: url("gala_demo/assets/background.jpeg");
                background-repeat: no-repeat;
                background-position: cover;
            }
        """)
        layout = QHBoxLayout()
        central_widget.setLayout(layout)

        # coordinate system initialization 
        self.sandbox_fig, self.ax = plt.subplots()
        self.sandbox_fig.patch.set_facecolor('none')
        self.ax.set_facecolor('none')
        self.ax.axis('off')
        self.sandbox_fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.sandbox_canvas = FigureCanvas(self.sandbox_fig)
        self.sandbox_canvas.setFixedWidth(1050)  
        self.sandbox_canvas.setFixedHeight(1050)
        self.sandbox_canvas.setStyleSheet("background-color:transparent;")
        layout.addWidget(self.sandbox_canvas)

        side_layout = QVBoxLayout()
        side_layout.setSpacing(5)

        # --- TOP IMAGES, Dolya logos ---
        dolya_label = QLabel()
        pixmap = QPixmap("gala_demo/assets/dolya.png")
        dolya_pixmap = pixmap.scaledToWidth(150)
        dolya_label.setPixmap(dolya_pixmap)
        dolya_wrapper = QHBoxLayout()
        dolya_wrapper.addStretch()
        dolya_wrapper.addWidget(dolya_label)
        dolya_wrapper.addStretch()
        dolya_container = QWidget()
        dolya_layout = QHBoxLayout()
        dolya_layout.setContentsMargins(298, 0, 0, 0)
        dolya_layout.addWidget(dolya_label)
        dolya_container.setLayout(dolya_layout)

        side_layout.addSpacing(40) 
        side_layout.addWidget(dolya_container)
        side_layout.addSpacing(10) 

        dtext_label = QLabel()
        pixmap = QPixmap("gala_demo/assets/dolya_text.png")
        dtext_pixmap = pixmap.scaledToWidth(500)
        dtext_label.setPixmap(dtext_pixmap)
        dtext_wrapper = QHBoxLayout()
        dtext_wrapper.addStretch() 
        dtext_wrapper.addWidget(dtext_label)
        dtext_wrapper.addStretch()
        dtext_container = QWidget()
        dtext_layout = QHBoxLayout()
        dtext_layout.setContentsMargins(140, 0, 0, 0) 
        dtext_layout.addWidget(dtext_label)
        dtext_container.setLayout(dtext_layout)
        side_layout.addWidget(dtext_container)
        side_layout.addSpacing(70) 

        # --- TEXT LABELS ---
        self.title_label = QLabel()
        self.title_label.setText(
            '<span style="font-family: Verdana; font-size: 37px; color: #ffffff;">'
            '<b>REAL-TIME MINE DETECTION DEMO</b>'
            '</span>'
            )
        side_layout.addWidget(self.title_label)
        side_layout.addSpacing(80) 

        self.mine_label = QLabel()
        self.mine_label.setText(
            '<span style="font-family: Verdana; font-size: 37px; color: #ffffff;">'
            '<b>State:</b> Sleeping'
            '</span>'
            )
        side_layout.addWidget(self.mine_label)

        self.gps_label = QLabel()
        self.gps_label.setText(
            '<span style="font-family: Verdana; font-size: 37px; color: #ffffff;">'
            '<b>GPS:</b> 60°11\'16" N 24°49\'28" E'
            '</span>'
        )
        side_layout.addWidget(self.gps_label)

        self.connection_label = QLabel()
        self.connection_label.setText(
            '<span style="font-family: Verdana; font-size: 37px; color: #ff8f8f;">'
            '<b>Jetson:</b> Waiting for connection...'
            '</span>'
        )
        side_layout.addWidget(self.connection_label)
        side_layout.addSpacing(80) 

        # --- BOTTOM IMAGE, Saab logo ---
        saab_label = QLabel()
        pixmap = QPixmap("gala_demo/assets/saab_text.png")
        scaled_pixmap = pixmap.scaledToWidth(650)
        saab_label.setPixmap(scaled_pixmap)
        saab_wrapper = QHBoxLayout()
        saab_wrapper.addStretch()
        saab_wrapper.addWidget(saab_label)
        saab_wrapper.addStretch()
        saab_container = QWidget()
        saab_layout = QHBoxLayout()
        saab_layout.setContentsMargins(30, 0, 0, 0)
        saab_layout.addWidget(saab_label)
        saab_container.setLayout(saab_layout)
        side_layout.addWidget(saab_container)
        side_layout.addSpacing(70) 

        layout.addLayout(side_layout)
        # -------- layout setup ends here --------

        # sandbox (perspective transformed)
        self.sandbox = mpimg.imread("gala_demo/assets/new_sandbox.png")
        self.sb_h, self.sb_w = self.sandbox.shape[:2]
        
        self.ax.imshow(self.sandbox, extent=(0, self.sb_w, 0, self.sb_h), zorder=0)
        margin_x = 5 * SCALING_RATIO; margin_y = 5 * SCALING_RATIO
        self.ax.set_xlim(-margin_x, self.sb_w + margin_x)
        self.ax.set_ylim(-margin_y, self.sb_h + margin_y)
        self.ax.set_aspect('equal')
        self.sandbox_canvas.draw()

        # pfm-1
        self.pfm_1 = mpimg.imread("gala_demo/assets/9n235.png")
        self.pfm_1 = resize_image(self.pfm_1, 12 * SCALING_RATIO)
        self.pfm_1_h, self.pfm_1_w = self.pfm_1.shape[:2]

        # 9n235
        self.n235 = mpimg.imread("gala_demo/assets/9n235.png")
        self.n235 = resize_image(self.n235, 24 * SCALING_RATIO)
        self.n235_h, self.n235_w = self.n235.shape[:2]

        # listen to the socket in the background
        self.data_queue = queue.Queue()
        Thread(target=listen_socket, args=(self.data_queue,), daemon=True).start()

        # timer to check queue and update
        self.timer = QTimer()
        self.timer.timeout.connect(self.fetch_data)
        self.timer.start(100)

    """ 
        Fetches data from the data queue and updates the Hiekkalaatikko window accordingly.
    """
    def fetch_data(self):
        try:
            while True:
                data = self.data_queue.get_nowait()
                if "__connection__" in data:
                    connected = data["__connection__"]
                    if connected:
                        time = data.get("time", "")
                        self.connection_label.setText(
                            f'<span style="font-family: Verdana; font-size: 37px; color: #a1e6b1;">'
                            f'<b>Jetson:</b> Connected - [{time}]'
                            f'</span>'
                        )
                    else:
                        self.connection_label.setText(
                            '<span style="font-family: Verdana; font-size: 37px; color: #ff8f8f;">'
                            '<b>Jetson:</b> Disconnected!'
                            '</span>'
                        )
                else:
                    state = data.get("jetson_state", "unknown")
                    if state == "detecting":
                        for mine in self.mines:
                            mine["artist"].remove()
                        self.mines.clear()

                        mine_results = data.get("mines", {})
                        if isinstance(mine_results, dict) and mine_results:
                            self.mine_label.setText(
                                '<span style="font-family: Verdana; font-size: 37px; color: #ffffff;">'
                                '<b>State:</b> Mines detected'
                                '</span>'
                            )

                            for mine_type, positions in mine_results.items():
                                if mine_type == "PFM-1":
                                    img = self.pfm_1
                                    img_w = self.pfm_1_w
                                    img_h = self.pfm_1_h
                                elif mine_type == "9N235":
                                    img = self.n235
                                    img_w = self.n235_w
                                    img_h = self.n235_h
                                else:
                                    # unknown mine type, skip
                                    continue
                                
                                M = np.loadtxt("gala_demo/assets/perspective_matrix.txt")
                                for (x, y) in positions:
                                    pt = np.array([x, y, 1], dtype=np.float32)
                                    transformed = M @ pt
                                    x_new = transformed[0] / transformed[2]
                                    y_new = transformed[1] / transformed[2]
                                    y_new_flipped = self.sb_h - y_new
                                    self.plot_mine(x_new, y_new_flipped, img, img_w, img_h)
                        else:
                            self.mine_label.setText(
                                '<span style="font-family: Verdana; font-size: 37px; color: #ffffff;">'
                                '<b>State:</b> No mines detected'
                                '</span>'
                            )


                        self.sandbox_canvas.draw_idle()
                        
                    elif state == 'uploading':
                        self.mine_label.setText(
                            '<span style="font-family: Verdana; font-size: 37px; color: #ffffff;">'
                            '<b>State:</b> Uploading'
                            '</span>'
                        )
                    else:
                        self.mine_label.setText(
                            '<span style="font-family: Verdana; font-size: 37px; color: #ffffff;">'
                            '<b>State:</b> Sleeping'
                            '</span>'
                        )
                self.update_label(data)
        except queue.Empty:
            pass
        
    """ 
        Plots a mine with the given specifications.
    """
    def plot_mine(self, x0: float, y0: float, image, img_w, img_h):
        artist = self.ax.imshow(
            image,
            extent=(x0, x0 + img_w, y0, y0 + img_h),
            zorder=1
        )
        self.mines.append({"artist": artist, "x": x0, "y": y0})

    """ 
        Updates the label for the Jetson connection.
    """
    def update_label(self, data):
        time = data.get("time", "")
        self.connection_label.setText(
            f'<span style="font-family: Verdana; font-size: 37px; color: #a1e6b1;">'
            f'<b>Jetson:</b> Connected <br>{time}'
            f'</span>'
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Hiekkalaatikko()
    window.show()
    sys.exit(app.exec_())
