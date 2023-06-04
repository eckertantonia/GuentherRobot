#!/usr/bin/env python3
import curses
from gpiozero import Robot
import cv2
import imutils

def remoteControl():

    robot = Robot(left=(5,6), right=(13,19))

    neutral = cv2.imread("ext_code/img/neutral.jpg")
    neutral = imutils.resize(neutral, width=500, height=500)

    happy = cv2.imread("ext_code/img/angry.jpg")
    happy = imutils.resize(happy, width=500, height=500)

    actions = {
        curses.KEY_UP:      robot.forward,
        curses.KEY_DOWN:    robot.backward,
        curses.KEY_LEFT:    robot.left,
        curses.KEY_RIGHT:   robot.right,
    }

    def main(window):
        next_key = None
        while True:
            curses.halfdelay(1)
            if next_key is None:
                key = window.getch()
                
            else:
                key = next_key
                next_key = None
            if key != -1:
                if key == ord('w'):
                    happy = cv2.imread("ext_code/img/angry.jpg")
                    happy = imutils.resize(happy, width=500, height=500)
                    cv2.imshow("Frame", happy)
                    continue
                elif key == ord('e'):
                    neutral = cv2.imread("ext_code/img/neutral.jpg")
                    neutral = imutils.resize(neutral, width=500, height=500)
                    cv2.imshow("Frame", neutral)
                    continue

                # KEY PRESSED
                curses.halfdelay(3)
                action = actions.get(key)
                if action is not None:
                    action()
                next_key = key
                while next_key == key:
                    next_key = window.getch()
                # KEY RELEASED
                robot.stop()

    curses.wrapper(main)
