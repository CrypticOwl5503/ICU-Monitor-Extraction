from pyimagesearch import config
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.layers import UpSampling2D, Concatenate
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Flatten, Dense, Input
from tensorflow.keras.applications import MobileNetV2



BASE_PATH="dataset"
IMAGES_PATH=os.path.sep.join([BASE_PATH, "images"])
ANNOTS_PATH=os.path.sep.join([BASE_PATH, "airplanes.csv"])

BASE_OUTPUT="output"

MODEL_PATH=os.path.sep.join([BASE_OUTPUT, "detector.h5"])
PLOT_PATH=os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_FILENAMES=os.path.sep.join([BASE_OUTPUT, "test_images.txt"])

INIT_LR=1e-4
NUM_EPOCHS=30
BATCH_SIZE=32


print("[INFO] loading dataset...")
rows=open(ANNOTS_PATH).read().strip().split("\n")

data=[]
targets=[]
filenames=[]

for row in rows[1:]:
    row=row.split(",")
    (filename, x1, y1, x2, y2, x3, y3, x4, y4) = row
    
    imagePath=os.path.sep.join([IMAGES_PATH, filename])
    image=cv2.imread(imagePath)
    (h, w)=(720, 1280)
    
    x1=float(x1)/w
    y1=float(y1)/h
    x2=float(x2)/w
    y2=float(y2)/h
    x3=float(x3)/w
    y3=float(y3)/h
    x4=float(x4)/w
    y4=float(y4)/h

    
    
    image=load_img(imagePath, target_size=(224,224))
    image=img_to_array(image)
    
    data.append(image)
    targets.append((x1, y1, x2, y2, x3, y3, x4, y4))
    filenames.append(filename)
    
data=np.array(data, dtype="float32") / 255.0
targets=np.array(targets, dtype="float32")

split=train_test_split(data, targets, filenames, test_size=0.10, random_state=42)


(trainImages, testImages)=split[:2]
(trainTargets, testTargets)=split[2:4]
(trainFilenames, testFilenames)=split[4:]


print("[INFO] saving testing filenames...")
f=open(TEST_FILENAMES, "w")
f.write("\n".join(testFilenames))
f.close()





vgg=VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))


vgg.trainable=False

def corner_net(input_shape=(224, 224, 3)):
    # Backbone network (you can choose a backbone architecture)
    backbone = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape))
    
    # Replace your VGG model with the backbone model
    backbone.trainable = False

    # CornerNet Head
    conv1 = Conv2D(256, (3, 3), padding='same')(backbone.layers[-1].output)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    
    conv2 = Conv2D(128, (3, 3), padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    
    conv3 = Conv2D(64, (3, 3), padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    
    conv4 = Conv2D(32, (3, 3), padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    
    conv5 = Conv2D(16, (3, 3), padding='same')(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    
    # Flatten the output
    flatten = Flatten()(conv5)
    
    # Predict 8 values for 4 pairs of coordinates
    coordinates = Dense(8, activation='sigmoid', name='coordinates')(flatten)
    
    # Create the CornerNet model
    model = Model(inputs=backbone.input, outputs=coordinates)

    return model

# Create the CornerNet model with ResNet-50 backbone
model = corner_net()


opt = Adam(lr=INIT_LR)
model.compile(loss='mse', optimizer=opt)
print(model.summary())


print("{INFO} training bounding box regressor....")
H=model.fit(trainImages, trainTargets, validation_data=(testImages, testTargets), batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1)


N=NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,N), H.history["val_loss"], label="val_loss")
plt.title("Bounding box regression loss on training set")
plt.xlabel("epoch no.")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(PLOT_PATH)
    
model.save(MODEL_PATH)