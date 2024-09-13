
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from glob import glob
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)
from tensorflow.keras.utils import CustomObjectScope
from tqdm import tqdm

from metrics import dice_coef, dice_loss, iou

H = 512
W = 512


""" Creating a directory """
def create_dir(path):
    """Create a directory."""
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except FileExistsError as e:
        pass

def load_data(path):
    x0 = sorted(glob(os.path.join(path, "images", "*.jpg")))
    x1 = sorted(glob(os.path.join(path, "color_masks", "*.jpg")))
    y = sorted(glob(os.path.join(path, "masks", "*.jpg")))
    return x0, x1, y

def save_results(image, color_image, mask, y_pred, save_image_path):
    ## i - m - y
    line = np.ones((H, 10, 3)) * 128

    """ Mask """
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)

    """ Predicted Mask """
    y_pred = np.expand_dims(y_pred, axis=-1)    ## (512, 512, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)  ## (512, 512, 3)
    y_pred = y_pred * 255

    # cat_images = np.concatenate([image, line, color_image, line, mask, line, y_pred], axis=1)
    cat_images = np.concatenate([image, line, mask, line, y_pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    # create_dir("results_1")
    save_image_path = os.path.join("unet_code1", "results1", "test_data", '512x512')
    create_dir(save_image_path)

    """ Loading model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("unet_code1/save_model_1/model.keras", safe_mode=False)

    """ Load the dataset """
    test_path = os.path.join("unet_code1", "data", "test_data", "512x512")
    result_path = os.path.join("unet_code1", "results1", "test_data", '512x512')
    # test_path = os.path.join("data", "test_data", "512x512")

    test_x0, test_x1, test_y = load_data(test_path)
    print(f"Test: {len(test_x0)} - {len(test_x1)} - {len(test_y)}")

    """ Evaluation and Prediction """
    SCORE = []
    for x0, x1, y, in tqdm(zip(test_x0, test_x1, test_y), total=len(test_x0)):
        """ Extract the name """
        # save_image_path = os.path.join("result_final", "test_data", '512x512', f"{name}")
        name = x0.split("/")[-1]

        """ Reading the image """
        image = cv2.imread(x0, cv2.IMREAD_COLOR + cv2.IMREAD_IGNORE_ORIENTATION)
        x0 = image/255.0
        x0 = np.expand_dims(x0, axis=0)

        """ Reading the image """
        color_image = cv2.imread(x1, cv2.IMREAD_COLOR + cv2.IMREAD_IGNORE_ORIENTATION)
        x1 = image/255.0
        x1 = np.expand_dims(x1, axis=0)

        """ Reading the mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        y = mask/255.0
        y = y > 0.5
        y = y.astype(np.int32)

        """ Prediction """
        y_pred = model.predict(x0)[0]
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(np.int32)

        """ Saving the prediction """

        
        save_image_path = os.path.join(result_path, f'{name}')
        save_results(image, color_image, mask, y_pred, save_image_path)

        """ Flatten the array """
        y = y.flatten()
        y_pred = y_pred.flatten()

        """ Calculating the metrics values """
        acc_value = accuracy_score(y, y_pred)
        f1_value = f1_score(y, y_pred, labels=[0, 1], average="binary", zero_division=1)
        jac_value = jaccard_score(y, y_pred, labels=[0, 1], average="binary", zero_division=1)
        recall_value = recall_score(y, y_pred, labels=[0, 1], average="binary", zero_division=1)
        precision_value = precision_score(y, y_pred, labels=[0, 1], average="binary", zero_division=1)
        SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value])

    """ Metrics values """
    score = [s[1:]for s in SCORE]
    score = np.mean(score, axis=0)
    print(f"Accuracy: {score[0]:0.5f}")
    print(f"F1: {score[1]:0.5f}")
    print(f"Jaccard: {score[2]:0.5f}")
    print(f"Recall: {score[3]:0.5f}")
    print(f"Precision: {score[4]:0.5f}")

    df = pd.DataFrame(SCORE, columns=["Image", "Accuracy", "F1", "Jaccard", "Recall", "Precision"])
    df.to_csv(f"{result_path}/d_evel_test_score.csv")
