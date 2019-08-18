import cv2
from mss import mss
import numpy as np
sct = mss()
while True:
    ACTIVATION_RANGE = 300
    Wd, Hd = sct.monitors[1]["width"], sct.monitors[1]["height"]
    origbox = (int(Wd / 2 - ACTIVATION_RANGE / 2),
               int(Hd / 2 - ACTIVATION_RANGE / 2),
               int(Wd / 2 + ACTIVATION_RANGE / 2),
               int(Hd / 2 + ACTIVATION_RANGE / 2))
    img = sct.grab(origbox)
    im = np.array(img)
    frame = cv2.UMat(im)
    frame = cv2.UMat(frame)
    cv2.imshow('image', frame)
    k = cv2.waitKey(0)
    if k == 27:  # wait for ESC key to exit
        cv2.destroyAllWindows()