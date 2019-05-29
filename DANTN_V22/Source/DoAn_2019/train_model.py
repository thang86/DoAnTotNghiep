from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from utilities.nn.cnn import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="đường dần tới dữ liệu đầu vào của khuôn mặt")
ap.add_argument("-m", "--model", required=True,
                help="đường dẫn tới mô hình đầu ra.")
args = vars(ap.parse_args())

# Initialize the list of data and labels
data = []
labels = []


for image_path in sorted(list(paths.list_images(args["dataset"]))):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=28)
    image = img_to_array(image)
    data.append(image)

    label = image_path.split(os.path.sep)[-3]
    label = "smiling" if label == "positives" else "not_smiling"
    labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)


class_totals = labels.sum(axis=0)
class_weight = class_totals.max() / class_totals

(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

print("[INFO]: Compiling model....")
model = LeNet.build(width=28, height=28, depth=1, classes=2)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

print("[INFO]: Training....")
H = model.fit(train_x, train_y, validation_data=(test_x, test_y), class_weight=class_weight,
              batch_size=64, epochs=15, verbose=1)

print("[INFO]: Evaluating....")
predictions = model.predict(test_x, batch_size=64)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

print("[INFO]: Serializing network....")
model.save(args["model"])


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 15), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 15), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 15), H.history["acc"], label="acc")
plt.plot(np.arange(0, 15), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
