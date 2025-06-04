from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306
from PIL import Image, ImageDraw, ImageFont, ImageOps
import time


""" 
    Implements the relevant functionalities of the I2C screen of the sensor module.
"""


SERIAL = i2c(port=1, address=0x3C)
WIDTH = 128; HEIGHT = 64
device = ssd1306(SERIAL, width=128, height=64)


""" 
   Displays an image on the screen with the possibility to invert the colors and crop the image.
"""
def display_image(path, invert=False, crop=None):
    img = Image.open(path).convert('1', dither=Image.NONE)
    if invert: img = ImageOps.invert(img.convert('L')).convert('1')

    if crop:
        left, right, top, bottom = crop
        img = img.crop((left, top, img.width - right, img.height - bottom))

    img = ImageOps.contain(img, (WIDTH, HEIGHT))
    img_padded = Image.new('1', (WIDTH, HEIGHT), color=0)
    offset = ((WIDTH - img.width) // 2, (HEIGHT - img.height) // 2)
    img_padded.paste(img, offset)

    device.display(img_padded)


""" 
    Displays a progress bar with the given text on the screen for a given duration.
"""
def draw_progress_bar(duration, text, bar_event=False):
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 9)
    text_y = 16; text_box = font.getbbox(text)
    text_width = text_box[2] - text_box[0]; text_height = text_box[3] - text_box[1]

    bar_width = int(WIDTH * 0.7); bar_height = 8
    bar_x = (WIDTH - bar_width) // 2; bar_y = text_y + text_height + 12
    blob_width = int(bar_width * 0.2); blob_speed = 4
    radius = 3

    if bar_event:
        while bar_event.is_set():
            for offset in range(0, bar_width + blob_width, blob_speed):
                img = Image.new('1', (WIDTH, HEIGHT), color=0)
                draw = ImageDraw.Draw(img)

                text_x = (WIDTH - text_width) // 2
                draw.text((text_x, text_y), text, font=font, fill=255)

                bar_top = bar_y
                bar_bottom = bar_top + bar_height
                draw.rounded_rectangle(
                    (bar_x, bar_top, bar_x + bar_width, bar_bottom), radius=radius, outline=255, fill=0
                )

                x0 = bar_x + (offset % (bar_width + blob_width)) - blob_width
                x1 = x0 + blob_width
                if x1 > bar_x:
                    draw.rounded_rectangle(
                        (max(bar_x, x0), bar_top, min(bar_x + bar_width, x1), bar_bottom), radius=radius, fill=255
                    )
                device.display(img)
    else:
        start_time = time.time()
        while time.time() - start_time < duration:
            for offset in range(0, bar_width + blob_width, blob_speed):
                img = Image.new('1', (WIDTH, HEIGHT), color=0)
                draw = ImageDraw.Draw(img)

                text_x = (WIDTH - text_width) // 2
                draw.text((text_x, text_y), text, font=font, fill=255)

                bar_top = bar_y
                bar_bottom = bar_top + bar_height
                draw.rounded_rectangle(
                    (bar_x, bar_top, bar_x + bar_width, bar_bottom), radius=radius, outline=255, fill=0
                )

                x0 = bar_x + (offset % (bar_width + blob_width)) - blob_width
                x1 = x0 + blob_width
                if x1 > bar_x:
                    draw.rounded_rectangle(
                        (max(bar_x, x0), bar_top, min(bar_x + bar_width, x1), bar_bottom), radius=radius, fill=255
                    )

                device.display(img)
    display_image('media/saab_text_logo.png', invert=True, crop=(5, 5, 40, 50))


""" 
    Displays a given text on the screen.
"""
def display_text(text):
    img = Image.new('1', (WIDTH, HEIGHT), color=0)
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 9)
    text_y = 16; text_box = font.getbbox(text)
    text_width = text_box[2] - text_box[0]; text_height = text_box[3] - text_box[1]
    text_x = (WIDTH - text_width) // 2
    draw.text((text_x, text_y), text, font=font, fill=255)

    device.display(img)