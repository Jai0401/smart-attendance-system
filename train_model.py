# USAGE
# python3 train_model.py --recognizer output/recognizer.joblib --le output/le.joblib

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from imutils import paths
import argparse
import imutils
import cv2
import joblib
import os
import numpy as np


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--recognizer", required=True,
    help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
    help="path to output label encoder")
args = vars(ap.parse_args())

# load serialized face detector
print("Loading Face Detector...")
protoPath = "face_detection_model/deploy.prototxt"
modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load serialized face embedding model
print("Loading Face Recognizer...")
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# grab the paths to the input images in our dataset
print("Quantifying Faces...")
imagePaths = list(paths.list_images("dataset"))

# initialize our lists of extracted facial embeddings and corresponding people names
knownEmbeddings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    if (i % 50 == 0):
        print("Processing image {}/{}".format(i, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    # load the image, resize it, and perform face detection
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply face detection
    detector.setInput(imageBlob)
    detections = detector.forward()

    # check if at least one face was found
    if len(detections) > 0:
        # we're making the assumption that each image has only ONE face, so find the bounding box with the largest probability
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        # ensure that the detection with the largest probability also means our minimum probability test (thus helping filter out weak detections)
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI and grab the ROI dimensions
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW >= 20 and fH >= 20:
                # construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                    (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # add the name of the person + corresponding face embedding to their respective lists
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(knownNames)

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(knownEmbeddings, labels)

# write the actual face recognition model to disk
joblib.dump(recognizer, args["recognizer"])

# write the label encoder to disk
joblib.dump(le, args["le"])
