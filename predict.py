from pyimagesearch import config
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import cv2
import os
import csv

# Define the directory containing images
IMAGE_DIRECTORY = "D:/Code/Inter IIT PS/vs code/dataset/images2"

BASE_OUTPUT = "output"
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.h5"])
CSV_OUTPUT_PATH = os.path.sep.join([BASE_OUTPUT, "image_coordinates.csv"])

INIT_LR = 1e-4
NUM_EPOCHS = 10
BATCH_SIZE = 32

def load_image_and_get_coordinates(imagePath, model):
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    preds = model.predict(image)[0]
    (x1, y1, x2, y2, x3, y3, x4, y4) = preds
    x1 = (x1 * 1280)
    y1 = (y1 * 720)
    x2 = (x2 * 1280)
    y2 = (y2 * 720)
    x3 = (x3 * 1280)
    y3 = (y3 * 720)
    x4 = (x4 * 1280)
    y4 = (y4 * 720)
    return x1, y1, x2, y2, x3, y3, x4, y4

print("[INFO] Loading object detector...")
model = load_model(MODEL_PATH)

with open(CSV_OUTPUT_PATH, "w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["ImageName", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"])

    for filename in os.listdir(IMAGE_DIRECTORY):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            imagePath = os.path.join(IMAGE_DIRECTORY, filename)
            x1, y1, x2, y2, x3, y3, x4, y4 = load_image_and_get_coordinates(imagePath, model)
            csv_writer.writerow([filename, x1, y1, x2, y2, x3, y3, x4, y4])
