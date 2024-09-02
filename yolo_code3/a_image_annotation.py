import math
import os
import threading
from glob import glob

import cv2
import numpy as np
from PIL import Image, ImageDraw


def find_radius(center_x, center_y, point_x, point_y):
    """Finds the radius of a circle given its center and a point on its circumference."""
    # Calculate the difference in x and y coordinates
    dx = point_x - center_x
    dy = point_y - center_y
    # Use the Pythagorean theorem to find the radius
    radius = math.sqrt(dx**2 + dy**2)
    return int(radius)


def create_dir(path):
    """Create a directory."""
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except FileExistsError as e:
        pass


def create_mask(img, coords, thickness):
    ImageDraw.Draw(img).ellipse(
        (
            coords[0] - thickness,
            coords[1] - thickness,
            coords[0] + thickness,
            coords[1] + thickness,
        ),
        fill=255,
        outline=None,
    )


def get_coordinates_for_circle(line=list):
    temp_list = []
    for obj in line[-4:]:
        temp_list.append(int(float(obj.strip("\n"))))
    temp_list = np.asarray(temp_list, dtype=int)
    return np.mean([temp_list[0], temp_list[2]]), np.mean([temp_list[1], temp_list[3]])


def get_coordinates_for_circle2(line=list):
    temp_list = []
    for obj in line[-4:]:
        temp_list.append(int(float(obj.strip("\n"))))
    temp_list = np.asarray(temp_list, dtype=int)
    radius = find_radius(*temp_list)
    return temp_list[0], temp_list[1], radius


def get_coordinates_for_point(line=list):
    temp_list = []
    for obj in line[5:]:
        temp_list.append(round(int(float(obj.strip("\n")))))
    temp_list = np.asarray(temp_list, dtype=int)
    return temp_list[0], temp_list[1]


def mark_img(img_path: str, dest_dir=None):

    image_data = cv2.imread(img_path, cv2.IMREAD_COLOR + cv2.IMREAD_IGNORE_ORIENTATION)
    # image_data_gray = cv2.imread(
    #     img_path, cv2.IMREAD_GRAYSCALE + cv2.IMREAD_IGNORE_ORIENTATION
    # )

    info_path = img_path.replace(
        "ThickBloodSmears_150", "ThickBloodSmears_150/GT_updated"
    ).replace(".jpg", ".txt")

    full_img = os.path.join(dest_dir, "image", "_".join(img_path.split("/")[1:]))
    color_mask_img = os.path.join(dest_dir, "color_mask", "_".join(img_path.split("/")[1:]))
    mask_img = os.path.join(dest_dir, "mask", "_".join(img_path.split("/")[1:]))

    full_path = "/".join(full_img.split("/")[:-1])
    color_mask_path = "/".join(color_mask_img.split("/")[:-1])
    mask_path = "/".join(mask_img.split("/")[:-1])
    create_dir(full_path)
    create_dir(color_mask_path)
    create_dir(mask_path)

    if os.path.exists(info_path):
        info = open(info_path.replace(".jpg", ".txt"))
    else:
        return

    parasites = []
    wbcs = []
    image_size = []
    if os.path.exists(info_path):
        for line in info.readlines():
            if "Parasite" in line.strip().split(","):
                parasites.append(line.split(","))
            elif "White_Blood_Cell" in line.split(","):
                wbcs.append(line.split(","))
            else:
                image_size.append(line.strip().split(","))
    cv2.imwrite(full_img, image_data)

    for para in parasites:
        color = (0, 0, 0)
        thickness = 1
        start_point, end_point, radius = get_coordinates_for_circle2(para)
        cv2.circle(
            image_data,
            (round(start_point), round(end_point)),
            radius,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
    cv2.imwrite(color_mask_img, image_data)

    mimg = Image.new("L", tuple(np.array(image_size[0][1:], dtype=int)), 0)
    for para in parasites:
        color = (0, 0, 255)
        thickness = 1
        radius = 10
        start_point, end_point, radius = get_coordinates_for_circle2(para)
        create_mask(mimg, (start_point, end_point), radius)
    mimg.save(mask_img)


def list_files_recursive(path, file_type=".jpg"):
    files = []
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            files.extend(list_files_recursive(full_path))
        else:
            if file_type in full_path:
                files.append(full_path)
    return files


def get_images_path():
    image_path_list = []
    if not os.path.exists("ThickBloodSmears_150"):
        return "folder ThickBloodSmears_150 does not exist."
    # image_path_list = list_files_recursive("ThickBloodSmears_150")
    image_path_list = sorted(
        glob(os.path.join("ThickBloodSmears_150", "**", "*.jpg"), recursive=True)
    )
    return image_path_list


semaphore = threading.Semaphore(10)


def worker(image, dest_dir):
    with semaphore:
        mark_img(image, dest_dir)


def create_marked_images():
    x = get_images_path()[:1000]
    rng_train = round(len(x) * 0.8)
    rng_val = round(len(x) * 0.1)
    rng_test = round(len(x) * 0.1)
    for image in x[:rng_train]:
        t = threading.Thread(target=worker, args=(image, "data/train_data/full_size"))
        t.start()
    for image in x[rng_train:rng_train+rng_val]:
        t = threading.Thread(target=worker, args=(image, "data/val_data/full_size"))
        t.start()
    for image in x[rng_train+rng_val:]:
        t = threading.Thread(target=worker, args=(image, "data/test_data/full_size"))
        t.start()
    print("done")


if __name__ == "__main__":
    create_marked_images()
