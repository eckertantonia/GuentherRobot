#!/usr/bin/env python3

from gpiozero import Robot
from time import sleep

def face_movement_OK(guenther, num):
    if num == 1:
        guenther.forward(curve_left=1)
        sleep(10)
        guenther.stop()
        sleep(0.5)
        guenther.forward()
        sleep(4)
        guenther.stop()
        sleep(1)
    elif num == 2:
        guenther.forward(curve_right=1)
        sleep(10)
        guenther.stop()
        sleep(0.5)
        guenther.forward()
        sleep(4)
        guenther.stop()
        sleep(1)
    elif num == 3:
        guenther.forward()
        sleep(0.5)
        guenther.backward()
        sleep(0.5)
        guenther.forward()
        sleep(0.5)
        guenther.backward()
        sleep(0.5)
        guenther.stop()

def no_mask(guenther):
    guenther.backward()
    sleep(3)
    guenther.stop()
    sleep(1)
    guenther.forward(0.75)
    sleep(4)
    guenther.stop()
    sleep(0.5)

def r_move(guenther):
    for i in range(3):
        guenther.forward(curve_right=0.5)
        sleep(0.5)
        guenther.stop()
        sleep(0.5)

def l_move(guenther):
    for i in range(3):
        guenther.forward(curve_left=0.5)
        sleep(0.5)
        guenther.stop()
        sleep(0.5)

def pose_movement(guenther):
    guenther.forward()
    sleep(1)
    guenther.stop()
    sleep(1)

if __name__ == '__main__':
    # Guenther init
    guenther = Robot(left=(5,6), right=(13,19))

    # movement
    face_movement_OK(guenther, 1)
    face_movement_OK(guenther, 2)
    face_movement_OK(guenther, 3)
    no_mask(guenther)
    l_move(guenther)
    r_move(guenther)
    pose_movement(guenther)
