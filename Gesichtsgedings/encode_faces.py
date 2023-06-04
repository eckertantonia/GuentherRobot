'''
have a gander @:
https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/?_ga=2.138517491.1151549932.1640203147-1821662895.1640203147#download-the-code
'''

from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

# argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True, help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="hog", help="face detection mode to use: either 'hog' or 'cnn'")
args = vars(ap.parse_args())

# grab paths
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# init lists
knownEncodings = []
knownNames = []

# looooooop over img paths
for (i, imagePath) in enumerate(imagePaths):
    # extract person name from path
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    # load and convert image to BGR
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detect bounding box
    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])

    # compute facial embedding for face
    encodings = face_recognition.face_encodings(rgb, boxes)

    # loop over encodings
    for encoding in encodings:
        # add encoding + name to known names and encodings
        knownEncodings.append(encoding)
        knownNames.append(name)

# write to disk
print("[INFO] serializing encodings...")
data = {"encodings" : knownEncodings, "names" : knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()