#!/usr/bin/env python3

from gpiozero import Robot
from time import sleep

if __name__ == '__main__':
    guenther = Robot(left=(5,6), right=(13,19))
    print("[INFO] GUENTHER INITIALIZED")
    print("OGOGOGOGOGOGO")
    guenther.forward()
    sleep(3)
    print("STOP")
    guenther.stop()
    