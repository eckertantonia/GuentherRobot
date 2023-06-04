#!/usr/bin/env python3

import cv2
from time import sleep

im = cv2.imread("img/smile.png")
im2 = cv2.imread("img/unsmile.png")
# cv2.namedWindow('im', cv2.WINDOW_FULLSCREEN)
cv2.imshow('Frame', im)
cv2.imshow('Frame2', im2)
cv2.waitKey(0)
cv2.destroyWindow('Frame')

# test 
