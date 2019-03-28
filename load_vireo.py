from imutils import paths
import random
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
from keras.preprocessing.image import img_to_array
import os

def load_vireo_data(data_dir, image_size):

    # initialize the data and labels
    data = []
    labels = []
    imagePaths = sorted(list(paths.list_images(data_dir)))
    random.seed(230)
    random.shuffle(imagePaths)

    # the data for training and the remaining 20% for testing
    split = int(0.8 * len(imagePaths))
    train_filenames = imagePaths[:split]
    eval_filenames = imagePaths[split:]

    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (image_size,image_size))
        image = img_to_array(image)
        data.append(image)

        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)

    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    print("[INFO] data matrix: {:.2f}MB".format(
        data.nbytes / (1024 * 1000.0)))

    # partition the data into training and testing splits using 80% of
    # the data for training and the remaining 20% for testing
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

    return trainX, testX, trainY, testY

