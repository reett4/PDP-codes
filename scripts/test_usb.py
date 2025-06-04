import os
import shutil


""" 
    Utility functions that can be used for transferring data from the system over USB. These have
    been tested to work, but aren't used in the current implementation.
"""


def find_usb_drives_linux():
    user = os.getenv("USER")
    potential_mount_dirs = [f"/media/{user}", f"/run/media/{user}"]
    usb_paths = []

    for mount_dir in potential_mount_dirs:
        if os.path.isdir(mount_dir):
            for entry in os.listdir(mount_dir):
                full_path = os.path.join(mount_dir, entry)
                if os.path.ismount(full_path):
                    usb_paths.append(full_path)

    return usb_paths

def find_first_image(directory):
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
    for filename in os.listdir(directory):
        if filename.lower().endswith(image_extensions):
            return os.path.join(directory, filename)
    return None

def copy_image_to_usb():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = find_first_image(script_dir)

    if not image_path:
        print("no image file found in script directory")
        return

    usb_drives = find_usb_drives_linux()

    if not usb_drives:
        print("no USB drive detected")
        return

    for usb_path in usb_drives:
        dest_path = os.path.join(usb_path, os.path.basename(image_path))
        try:
            shutil.copy2(image_path, dest_path)
            print(f"copied {image_path} → {dest_path}")
        except Exception as e:
            print(f"failed to copy to {usb_path}: {e}")

if __name__ == "__main__":
    copy_image_to_usb()