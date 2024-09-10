import os
import tempfile

import boto3
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope

S3_BUCKET_NAME = os.environ["BUCKET_NAME"]
UNET0_MODEL_FILE = os.environ["UNET0_MODEL_FILE"]
UNET1_MODEL_FILE = os.environ["UNET1_MODEL_FILE"]
s3 = boto3.client("s3")


# smooth = 1e-15
H = 512
W = 512
smooth = 1e-15


def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x

    return tf.numpy_function(f, [y_true, y_pred], tf.float32)


def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def get_s3_object_to_buffer(bucket_name, object_key):
    response = s3.get_object(Bucket=bucket_name, Key=object_key)
    buffer = response["Body"].read()
    return buffer


def put_img_to_s3_object(bucket_name, object_key, image_data):
    s3.put_object(Bucket=bucket_name, Key=object_key, Body=image_data)


def get_img_public_url(bucket_name, object_key):
    result_url = s3.generate_presigned_url(
        "get_object", Params={"Bucket": bucket_name, "Key": object_key}, ExpiresIn=300
    )
    return result_url


def save_results(image, y_pred, save_image_path):
    ## i - m - y
    line = np.ones((H, 10, 3)) * 128

    """ Predicted Mask """
    y_pred = np.expand_dims(y_pred, axis=-1)  ## (512, 512, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)  ## (512, 512, 3)
    y_pred = y_pred * 255

    cat_images = np.concatenate([image, line, y_pred], axis=1)
    image_bytes = cv2.imencode(".jpg", cat_images)[1].tobytes()
    put_img_to_s3_object(S3_BUCKET_NAME, save_image_path, image_bytes)


def save_results_all(image, y_pred_unet0, y_pred_unet1, save_image_path):
    ## i - m - y
    line = np.ones((H, 10, 3)) * 128

    """ Predicted Mask """
    y_pred_unet0 = np.expand_dims(y_pred_unet0, axis=-1)  ## (512, 512, 1)
    y_pred_unet0 = np.concatenate([y_pred_unet0, y_pred_unet0, y_pred_unet0], axis=-1)  ## (512, 512, 3)
    y_pred_unet0 = y_pred_unet0 * 255

    """ Predicted Mask """
    y_pred_unet1 = np.expand_dims(y_pred_unet1, axis=-1)  ## (512, 512, 1)
    y_pred_unet1 = np.concatenate([y_pred_unet1, y_pred_unet1, y_pred_unet1], axis=-1)  ## (512, 512, 3)
    y_pred_unet1 = y_pred_unet1 * 255

    cat_images = np.concatenate([image, line, y_pred_unet0, line, y_pred_unet1], axis=1)
    image_bytes = cv2.imencode(".jpg", cat_images)[1].tobytes()
    put_img_to_s3_object(S3_BUCKET_NAME, save_image_path, image_bytes)


def validate_input(event):
    s3_input_file = event.get("s3_input_file", None)
    model_type = event.get("model_type", None)
    model_type_choices = ["unet0", "unet1", "all"]
    if s3_input_file and model_type:
        if model_type not in model_type_choices:
            status = {
                "Error": f"Invalid model_type value choices are {model_type_choices}"
            }
        else:
            status = "success"
    else:
        status = {
            "Error": f"Missing Input 's3_input_file' and 'model_type' (choices are {model_type_choices})"
        }
    return status, s3_input_file, model_type


def create_dir(path):
    """Create a directory."""
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except FileExistsError as e:
        pass


def save_or_get_model(model_name, model_path):
    create_dir("/tmp/save_model")
    local_model_file = f"/tmp/save_model/model-{model_name}.keras"
    if not os.path.isfile(local_model_file):
        object_buffer = get_s3_object_to_buffer(S3_BUCKET_NAME, model_path)
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir="/tmp/save_model", prefix="model-", suffix=".keras")
        temp_file.write(object_buffer)
        os.rename(temp_file.name, local_model_file)
    return local_model_file


def get_unet0_output(s3_input_file, save=True):
    result_file = s3_input_file.split("/")[-1]
    result_file = f"lambda_api_predict/unet0/{result_file}"
    x0 = np.frombuffer(get_s3_object_to_buffer(S3_BUCKET_NAME, s3_input_file), np.uint8)

    """ Reading the image """
    image = cv2.imdecode(x0, cv2.IMREAD_COLOR + cv2.IMREAD_IGNORE_ORIENTATION)
    x0 = image / 255.0
    x0 = np.expand_dims(x0, axis=0)


    # object_buffer = get_s3_object_to_buffer(S3_BUCKET_NAME, UNET0_MODEL_FILE)
    # temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".keras")
    # temp_file.write(object_buffer)
    local_model_file = save_or_get_model("unet0", UNET0_MODEL_FILE)

    y_pred = None
    with CustomObjectScope(
        {"iou": iou, "dice_coef": dice_coef, "dice_loss": dice_loss}
    ):
        model = tf.keras.models.load_model(local_model_file, safe_mode=False)

        """ Prediction """
        y_pred = model.predict(x0)[0]
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(np.int32)

    if save:
        """ Saving the prediction """
        save_results(image, y_pred, result_file)
        result_image_url = get_img_public_url(S3_BUCKET_NAME, result_file)
        return result_image_url
    else:
        return y_pred


def get_unet1_output(s3_input_file, save=True):
    result_file = s3_input_file.split("/")[-1]
    result_file = f"lambda_api_predict/unet1/{result_file}"
    x0 = np.frombuffer(get_s3_object_to_buffer(S3_BUCKET_NAME, s3_input_file), np.uint8)

    """ Reading the image """
    image = cv2.imdecode(x0, cv2.IMREAD_COLOR + cv2.IMREAD_IGNORE_ORIENTATION)
    x0 = image / 255.0
    x0 = np.expand_dims(x0, axis=0)


    # object_buffer = get_s3_object_to_buffer(S3_BUCKET_NAME, UNET1_MODEL_FILE)
    # temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".keras")
    # temp_file.write(object_buffer)
    local_model_file = save_or_get_model("unet1", UNET1_MODEL_FILE)
    
    y_pred = None
    with CustomObjectScope(
        {"iou": iou, "dice_coef": dice_coef, "dice_loss": dice_loss}
    ):
        model = tf.keras.models.load_model(local_model_file, safe_mode=False)
        
        """ Prediction """
        y_pred = model.predict(x0)[0]
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(np.int32)

    if save:
        """ Saving the prediction """
        save_results(image, y_pred, result_file)
        result_image_url = get_img_public_url(S3_BUCKET_NAME, result_file)
        return result_image_url
    else:
        return y_pred


def get_all_output(s3_input_file, save=True):
    result_file_name = s3_input_file.split("/")[-1]
    result_file = f"lambda_api_predict/all/{result_file_name}"
    x0 = np.frombuffer(get_s3_object_to_buffer(S3_BUCKET_NAME, s3_input_file), np.uint8)

    """ Reading the image """
    image = cv2.imdecode(x0, cv2.IMREAD_COLOR + cv2.IMREAD_IGNORE_ORIENTATION)

    y_pred_unet0 = get_unet0_output(s3_input_file, save=False)
    y_pred_unet1 = get_unet1_output(s3_input_file, save=False)
    save_results_all(image, y_pred_unet0, y_pred_unet1, result_file)
    result_image_url = get_img_public_url(S3_BUCKET_NAME, result_file)
    return result_image_url


def predict_output(s3_input_file, model_type):
    if model_type == "unet0":
        result_image_url = get_unet0_output(s3_input_file)
    elif model_type == "unet1":
        result_image_url = get_unet1_output(s3_input_file)
    elif model_type == "all":
        result_image_url = get_all_output(s3_input_file)
    return result_image_url


def lambda_handler(event, context):
    print("lambda_handler called")

    input_status, s3_input_file, model_type = validate_input(event)

    if input_status == "success":
        result_image_url = predict_output(s3_input_file, model_type)
        return {
            "statusCode": 200,
            "body": {
                "result_image_url": result_image_url,
            },
            "headers": {
                "Angess-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "OPTIONS, POST, GET",
            },
        }
    else:
        return {
            "statusCode": 400,
            "body": input_status,
            "headers": {
                "Angess-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "OPTIONS, POST, GET",
            },
        }
        
