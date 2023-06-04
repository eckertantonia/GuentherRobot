from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
from ext_code.movement import face_movement_OK
from ext_code.movement import no_mask
import random

# arg parser
#ap = argparse.ArgumentParser()
#ap.add_argument("-c", "--cascade", required=True, help="path where face cascade resides")
#ap.add_argument("-e", "--encodings", required=True, help="path to serialized db with facial encodings")
#args = vars(ap.parse_args())

# load known faces
""" print("[INFO] loading encodings + face detector...")
data = pickle.loads(open("data/enc.pickle", "rb").read())
detector = cv2.CascadeClassifier("data/cascade.xml")

# init video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start() einkommentieren und darueber auskommentieren wenn auf pi (theoretisch)
time.sleep(2.0) """

# start fps counter
#fps = FPS().start()

# putt machen
def halt(vs):
    # cleanup
    cv2.destroyAllWindows()
    vs.stop()

# eigene func, die den ganzen spass ausfuehren tut
def gibIhm(guenther):
    print("[INFO] loading encodings + face detector...")
    data = pickle.loads(open("data/enc.pickle", "rb").read())
    detector = cv2.CascadeClassifier("data/cascade.xml")
    name = "no_mask"
    face_counter = 0
    noMask_counter = 0
    NO_DETECTION_THRESHOLD = 8
    DETECTION_THRESHOLD = 5


    # init video stream
    print("[INFO] starting video stream...")
    # TODO: Hier VideoStream aus Main uebergeben und dann starten
    #vs = vid_stream
    vs = VideoStream(src=0).start()
    #vs.start()
    # vs = VideoStream(src=0).start()
    # vs = VideoStream(usePiCamera=True).start() einkommentieren und darueber auskommentieren wenn auf pi (theoretisch)
    time.sleep(2.0)
    # looooooop over video frames
    while True:
        # grab and resize frame
        frame = vs.read()
        frame = imutils.resize(frame, width=500, height = 500)

        # convert frame to grayscale (face detection)
        # convert frame to rgb (face recognition)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # detect face in grayscale frame
        rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

        # reorder bounding boxes
        boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

        # compute facial embeddings for each face in bounding box
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

        rand = random.randrange(1,4)
        if rand == 1:
            emotion = cv2.imread("ext_code/img/neutral.jpg")
            emotion = imutils.resize(emotion, width=500, height=500)
        elif rand == 2:
            emotion = cv2.imread("ext_code/img/lookLeft.jpg")
            emotion = imutils.resize(emotion, width=500, height=500)
        elif rand == 3:
            emotion = cv2.imread("ext_code/img/lookRight.jpg")
            emotion = imutils.resize(emotion, width=500, height=500)

        # loop over facial encodings and check for matches
        for encoding in encodings:
            # attempt match for known encodings
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "no_mask"

            # check if we found a match
            if True in matches:
                # find indexes of all matched faces and create dict with total
                # number of times a face was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over matched indexes
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine recognized face with largest number of votes
                name = max(counts, key=counts.get)

            # update list of names
            names.append(name)

        # looooop over recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # draw predicted face name on image
            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left,y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

            # NEUNEUNEU
            if name == "no_mask":
                emotion = cv2.imread("ext_code/img/angry.jpg")
                emotion = imutils.resize(emotion, width=500, height=500)
            elif name == "has_mask":
                emotion = cv2.imread("ext_code/img/happy.jpg")
                emotion = imutils.resize(emotion, width=500, height=500)
            else:
                rand = random.randrange(1,4)
                if rand == 1:
                    emotion = cv2.imread("ext_code/img/neutral.jpg")
                    emotion = imutils.resize(emotion, width=500, height=500)
                elif rand == 2:
                    emotion = cv2.imread("ext_code/img/lookLeft.jpg")
                    emotion = imutils.resize(emotion, width=500, height=500)
                elif rand == 3:
                    emotion = cv2.imread("ext_code/img/lookRight.jpg")
                    emotion = imutils.resize(emotion, width=500, height=500)

        # display img to screen
        cv2.imshow("Frame", emotion)
        key = cv2.waitKey(1) & 0xFF

        if name == "has_mask":
            face_counter += 1
            if face_counter > DETECTION_THRESHOLD:
                face_movement_OK(guenther)
                halt(vs)
                break
        elif name == "no_mask":
            noMask_counter += 1
            if noMask_counter > NO_DETECTION_THRESHOLD:
                no_mask(guenther)
                noMask_counter = 0
        else:
            face_counter = 0
            noMask_counter = 0

        # if 'q' pressed break loop
        if key == ord("q"):
            halt(vs)
            break


    # update fps
    #fps.update()

#print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
#print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# stop timer and fps info
#fps.stop()
