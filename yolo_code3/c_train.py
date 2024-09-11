
import os
from glob import glob

import cv2
import numpy as np
from sklearn.utils import shuffle
import yaml

from a_image_annotation import create_dir
# from model_1 import build_unet

W = 512
H = 512


def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def load_data(path):
    x = sorted(glob(os.path.join(path, "images", "*.jpg")))
    y = sorted(glob(os.path.join(path, "masks", "*.jpg")))
    return x, y

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR + cv2.IMREAD_IGNORE_ORIENTATION)
    x = x/255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = x/255.0
    x = x > 0.5
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x


def mask_to_polygons(mask,epsilon=1.0):
    contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) > 2:
           poly = contour.reshape(-1).tolist()
           if len(poly) > 4: #Ensures valid polygon
              polygons.append(poly)
    return polygons

def process_data(image_paths, mask_paths):
    annotations = []
    images = []
    image_id = 0
    ann_id = 0

    for img_path, mask_path in zip(image_paths, mask_paths):
        image_id += 1
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        # shutil.copy(img_path, os.path.join(output_images_dir, os.path.basename(img_path)))

        # Add image to the list
        images.append({
            "id": image_id,
            "img_path": img_path,
            "file_name": os.path.basename(img_path),
            "height": img.shape[0],
            "width": img.shape[1]
        })

        unique_values = np.unique(mask)
        for value in unique_values:
            if value == 0:  # Ignore background
                continue

            object_mask = (mask == value).astype(np.uint8) * 255
            polygons = mask_to_polygons(object_mask)

            for poly in polygons:
                ann_id += 1
                annotations.append({                
                    "image_id": image_id,
                    "category_id": 1,  # Only one category: Nuclei
                    "segmentation": [poly],
                })

    coco_input = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "Nuclei"}]
    }

    # # Convert COCO-like dictionary to YOLO format
    for img_info in coco_input["images"]:
        img_id = img_info["id"]
        img_ann = [ann for ann in coco_input["annotations"] if ann["image_id"] == img_id]
        img_w, img_h = img_info["width"], img_info["height"]

        if img_ann:
            img_path = img_info["img_path"]
            res = img_path.replace("/images/", "/masks_label/").replace(".jpg", ".txt")
            res_path = "/".join(res.split("/")[:-1])
            create_dir(res_path)
            with open(res, 'w') as file_object:
                for ann in img_ann:
                    current_category = ann['category_id'] - 1
                    polygon = ann['segmentation'][0]
                    normalized_polygon = [format(coord / img_w if i % 2 == 0 else coord / img_h, '.6f') for i, coord in enumerate(polygon)]
                    file_object.write(f"{current_category} " + " ".join(normalized_polygon) + "\n")




def process_data2(image_paths, mask_paths):
    annotations = []
    images = []
    image_id = 0
    ann_id = 0

    for img_path, mask_path in zip(image_paths, mask_paths):
        image_id += 1
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE+cv2.IMREAD_UNCHANGED)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        res = img_path.replace("-1-original.jpg", "-5-mask.txt")
        with open(res, 'w') as f:
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                x_center = (x + w / 2) / mask.shape[1]
                y_center = (y + h / 2) / mask.shape[0]
                width = w / mask.shape[1]
                height = h / mask.shape[0]
                f.write(f"0 {x_center} {y_center} {width} {height}\n")



def create_yaml(output_yaml_path, train_images_dir, val_images_dir, nc=1):
    # Assuming all categories are the same and there is only one class, 'Nuclei'
    names = ['BloodSmears']
    ABS_PATH = os.getcwd()

    # Create a dictionary with the required content
    yaml_data = {
        'names': names,
        'nc': nc,  # Number of classes
        'train': os.path.join(ABS_PATH, train_images_dir),
        'val': os.path.join(ABS_PATH, val_images_dir),
        'test': ''
    }

    # Write the dictionary to a YAML file
    with open(output_yaml_path, 'w') as file:
        yaml.dump(yaml_data, file, default_flow_style=False)





if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)


    # data_dir = 'Nuclei-Instance-Dataset'
    output_dir = 'yolo_code3/yolov8_l_dataset'
    create_dir(output_dir)

    train_path = os.path.join("yolo_code3", "data", "train_data", "512x512")
    train_img_paths, train_mask_paths = shuffling(*load_data(train_path))

    val_path = os.path.join("yolo_code3", "data", "val_data", "512x512")
    val_img_paths, val_mask_paths = shuffling(*load_data(val_path))


    print(f"train_img_paths: {len(train_img_paths)}:{train_img_paths[0]}")
    print(f"train_mask_paths: {len(train_mask_paths)}:{train_mask_paths[0]}")
    print(f"val_img_paths: {len(val_img_paths)}:{val_img_paths[0]}")
    print(f"val_mask_paths: {len(val_mask_paths)}:{val_mask_paths[0]}")

    # Process and save the data in YOLO format for training and validation
    # process_data(train_img_paths[:1], train_mask_paths[:1])
    process_data(train_img_paths, train_mask_paths)
    process_data(val_img_paths, val_mask_paths)
    
    output_yaml_path = os.path.join(output_dir, 'data.yaml')
    create_yaml(output_yaml_path, train_path, val_path)
