import tensorflow as tf
import numpy as np
import cv2
import os

# Path
saved_model_dir = "best_saved_model"
output_path = "best_true_int8.tflite"

# Representative dataset folder (IMPORTANT)
dataset_path = "dataset/images/train"

IMG_SIZE = 320

# Representative dataset generator
def representative_data_gen():

    image_files = os.listdir(dataset_path)

    for img_name in image_files[:200]:

        img_path = os.path.join(dataset_path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.0

        img = np.expand_dims(img, axis=0)

        yield [img]


# Converter
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.representative_dataset = representative_data_gen

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()

with open(output_path, "wb") as f:
    f.write(tflite_model)

print("TRUE INT8 model saved")
