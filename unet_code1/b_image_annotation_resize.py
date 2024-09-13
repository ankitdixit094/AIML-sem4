import math
import os
import threading
from glob import glob

import cv2
import numpy as np
from PIL import Image, ImageDraw

from a_image_annotation import create_dir

W = 512
H = 512


def load_data(path):
    x = sorted(glob(os.path.join(path, "images", "*.jpg")))
    y = sorted(glob(os.path.join(path, "masks", "*.jpg")))
    return x, y


def get_images_path(patten: str="**"):
    path = os.path.join("unet_code1", "data", "**", "full_size", f"{patten}", f"*.jpg")
    image_path_list = sorted(
        glob(
            path,
            recursive=True,
        )
    )
    return image_path_list


semaphore = threading.Semaphore(10)


def resize_img(img_path: str, dest_dir=None):
    x = img_path
    x1 = img_path.replace("/images/", "/color_masks/")
    y = img_path.replace("/images/", "/masks/")
    resize_img = img_path.replace("full_size", f"{dest_dir}")
    resize_img_color = resize_img.replace("/images/", "/color_masks/")
    resize_img_mask = resize_img.replace("/images/", "/masks/")
    x = cv2.imread(x, cv2.IMREAD_COLOR + cv2.IMREAD_IGNORE_ORIENTATION)
    x1 = cv2.imread(x1, cv2.IMREAD_COLOR + cv2.IMREAD_IGNORE_ORIENTATION)
    y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (W, H))
    x1 = cv2.resize(x1, (W, H))
    y = cv2.resize(y, (W, H))
    resize_img_path = "/".join(resize_img.split("/")[:-1])
    resize_img_color_path = "/".join(resize_img_color.split("/")[:-1])
    resize_img_mask_path = "/".join(resize_img_mask.split("/")[:-1])
    create_dir(resize_img_path)
    create_dir(resize_img_color_path)
    create_dir(resize_img_mask_path)
    cv2.imwrite(resize_img, x)
    cv2.imwrite(resize_img_color, x1)
    cv2.imwrite(resize_img_mask, y)


def worker(image, dest_dir):
    with semaphore:
        resize_img(image, dest_dir)


def resize_images():
    images = get_images_path(patten="images")
    for image in images:
        t = threading.Thread(target=worker, args=(image, "512x512"))
        t.start()
    print("done")


if __name__ == "__main__":
    resize_images()
