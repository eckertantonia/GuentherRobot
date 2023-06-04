#!/usr/bin/env python3

# Imports
from imutils.video import VideoStream
import ext_code.face_reg as piface
# import ext_code.pose_reg as pipose
from gpiozero import Robot
from time import sleep

# Main
if __name__ == '__main__':
    # Video Stream hier initialisiern und immer an die Funktionien durchreichen
    #vs = VideoStream(src=0)

    # Guenther
    print("na mal gucken...")
    guenther = Robot(left=(5,6), right=(13,19))
    #guenther.forward()
    #sleep(3)
    #guenther.stop()

    # Anweisungen
    print("na mal gucken...")
    # pipose.poseRegSchleife(guenther)
    # sleep(5)
    piface.gibIhm(guenther)
    sleep(5)
    print("[INFO] abbruch...")
