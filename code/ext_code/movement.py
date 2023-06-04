#!/usr/bin/env python3
from time import sleep
from gpiozero import Robot
import random

def pose_movement(guenther):
    guenther.forward()
    sleep(1)
    guenther.stop()

def face_movement_OK(guenther):
    rand = random.randrange(1,4)
    if rand == 1:
        guenther.forward(curve_left=1)
        sleep(10)
        guenther.stop()
        sleep(0.5)
        guenther.forward()
        sleep(4)
        guenther.stop()
        sleep(1)
    elif rand == 2:
        guenther.forward(curve_right=1)
        sleep(10)
        guenther.stop()
        sleep(0.5)
        guenther.forward()
        sleep(4)
        guenther.stop()
        sleep(1)
    elif rand == 3:
        guenther.forward()
        sleep(0.5)
        guenther.backward()
        sleep(0.5)
        guenther.forward()
        sleep(0.5)
        guenther.backward()
        sleep(0.5)

def r_move(guenther):
    guenther.forward(curve_right=0.5)
    sleep(0.5)
    guenther.stop()

def l_move(guenther):
    guenther.forward(curve_left=0.5)
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
