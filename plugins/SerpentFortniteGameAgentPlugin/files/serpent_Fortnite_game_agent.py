import time
import math
import numpy as np
from datetime import datetime
import serpent.utilities
from serpent.frame_transformer import FrameTransformer
import re

from serpent.enums import InputControlTypes
import random
from serpent.config import config
from serpent.frame_grabber import FrameGrabber
from serpent.game_agent import GameAgent
from serpent.input_controller import KeyboardKey

from serpent.machine_learning.reinforcement_learning.agents.ppo_agent import PPOAgent as SerpentPPO

import os

import pytesseract

from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
import pyautogui
import pynput
import ctypes

import cv2

from mss import mss

import _thread

SendInput = ctypes.windll.user32.SendInput
PUL = ctypes.POINTER(ctypes.c_ulong)
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

def HoldKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = pynput._util.win32.INPUT_union()
    ii_.ki = pynput._util.win32.KEYBDINPUT(0, hexKeyCode, 0x0008, 0, ctypes.cast(ctypes.pointer(extra), ctypes.c_void_p))
    x = pynput._util.win32.INPUT(ctypes.c_ulong(1), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = pynput._util.win32.INPUT_union()
    ii_.ki = pynput._util.win32.KEYBDINPUT(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.cast(ctypes.pointer(extra), ctypes.c_void_p))
    x = pynput._util.win32.INPUT(ctypes.c_ulong(1), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
def set_pos(x, y):
    x = 1 + int(x * 65536. / 1920.)
    y = 1 + int(y * 65536. / 1080.)
    extra = ctypes.c_ulong(0)
    ii_ = pynput._util.win32.INPUT_union()
    ii_.mi = pynput._util.win32.MOUSEINPUT(x, y, 0, (0x0001 | 0x8000), 0,
                                           ctypes.cast(ctypes.pointer(extra), ctypes.c_void_p))
    command = pynput._util.win32.INPUT(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))


def left_click():
    extra = ctypes.c_ulong(0)
    ii_ = pynput._util.win32.INPUT_union()
    ii_.mi = pynput._util.win32.MOUSEINPUT(0, 0, 0, 0x0002, 0, ctypes.cast(ctypes.pointer(extra), ctypes.c_void_p))
    x = pynput._util.win32.INPUT(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    extra = ctypes.c_ulong(0)
    ii_ = pynput._util.win32.INPUT_union()
    ii_.mi = pynput._util.win32.MOUSEINPUT(0, 0, 0, 0x0004, 0, ctypes.cast(ctypes.pointer(extra), ctypes.c_void_p))
    x = pynput._util.win32.INPUT(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
def hold_left_click():
    extra = ctypes.c_ulong(0)
    ii_ = pynput._util.win32.INPUT_union()
    ii_.mi = pynput._util.win32.MOUSEINPUT(0, 0, 0, 0x0002, 0, ctypes.cast(ctypes.pointer(extra), ctypes.c_void_p))
    x=pynput._util.win32.INPUT(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def release_left_click():
    extra = ctypes.c_ulong(0)
    ii_ = pynput._util.win32.INPUT_union()
    ii_.mi = pynput._util.win32.MOUSEINPUT(0, 0, 0, 0x0004, 0, ctypes.cast(ctypes.pointer(extra), ctypes.c_void_p))
    x=pynput._util.win32.INPUT(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

class SerpentFortniteGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play
        self.frame_handler_setups["PLAY"] = self.setup_play

    def setup_play(self):
        self.sct = mss()
        self.monitor = {"top": 994, "left": 815, "width": 25, "height": 20}
        self.input_mapping = {
            "KEY_W": [KeyboardKey.KEY_W],
            "KEY_A": [KeyboardKey.KEY_A],
            "KEY_S": [KeyboardKey.KEY_S],
            "KEY_D": [KeyboardKey.KEY_D],
            "KEY_SPACE": [KeyboardKey.KEY_SPACE],
            "KEY_C": [KeyboardKey.KEY_C],
            "KEY_1": [KeyboardKey.KEY_1],
            "KEY_2": [KeyboardKey.KEY_2],
            "KEY_3": [KeyboardKey.KEY_3]
        }
        self.game_inputs = [
            {
                "name": "CONTROLS",
                "control_type": InputControlTypes.DISCRETE,
                "inputs": self.input_mapping
            }
        ]
        self.agent = SerpentPPO(
            "Fortnite",
            input_shape=(100, 100),
            game_inputs=self.game_inputs
        )

        self.started_at = datetime.utcnow().isoformat()

        self.analytics_client.track(event_key="GAME_NAME", data={"name": "Fortnite"})


    def handle_play(self, game_frame, game_frame_pipeline):
        self.handle_data(game_frame, game_frame_pipeline)
        game_input = self.handle_data(game_frame, game_frame_pipeline)
        run_input(game_input)
    def handle_data(self, game_frame, game_frame_pipeline):
        hp_int = self._measure_actor_hp()
        try:
            if hp_int < 10:
                terminal = {
                    1: False
                }
            else:
                terminal = {
                    0: True
                }
        except:
            terminal = terminal = {
                1: False
            }
        self.agent.observe(reward=self.reward_ai(), terminal=terminal)

        frame_buffer = FrameGrabber.get_frames([0, 2, 4, 6], frame_type="PIPELINE")
        self.game_input = self.agent.generate_actions(frame_buffer)
        self.game_input = str(self.game_input)
        print(self.game_input)
        return self.game_input

    def _measure_actor_hp(self):
        img_health = self.sct.grab(self.monitor)
        img_health = Image.frombytes("RGB", img_health.size, img_health.bgra, "raw", "BGRX")
        img_health = img_health.resize((100, 80))
        text = pytesseract.image_to_string(img_health, config='digits')
        text = text.replace(' ', '')
        text = re.sub("\D", "", text)
        if self.num_there(text) is True:
            text = int(text)
            if text == 0:
                return 100
            elif type(text) == None:
                return 100
            else:
                return text
    def reward_ai(self):
        self.hp = self._measure_actor_hp()
        self.reward = 1
        try:
            self.reward = self.reward + self.hp
        except:
            self.reward = 1
        self.reward = int(self.reward)
        return self.reward

    def extract_game_area(self, frame_buffer):
        game_area_buffer = []
        for game_frame in frame_buffer.frames:
            game_area = serpent.cv.extract_region_from_image(
                game_frame.grayscale_frame,
                self.game.screen_regions["GAME_REGION"]
            )
            frame = FrameTransformer.rescale(game_area, 0.25)
            game_area_buffer.append(frame)

        return game_area_buffer
    def num_there(self, s):
        return any(i.isdigit() for i in s)
YOLO_DIRECTORY = "models"
weightsPath = os.path.sep.join([YOLO_DIRECTORY, "yolov3-tiny.weights"])
configPath = os.path.sep.join([YOLO_DIRECTORY, "yolov3-tiny.cfg"])
CONFIDENCE = 0.36
THRESHOLD = 0.22
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
labelsPath = os.path.sep.join([YOLO_DIRECTORY, "coco-dataset.labels"])
LABELS = open(labelsPath).read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")
sct = mss()
ACTIVATION_RANGE = 250
Wd, Hd = sct.monitors[1]["width"], sct.monitors[1]["height"]
origbox = (int(Wd / 2 - 250 / 2),
           int(Hd / 2 - ACTIVATION_RANGE / 2),
           int(Wd / 2 + 250 / 2),
           int(Hd / 2 + ACTIVATION_RANGE / 2))
def do_third_input(game_input):
    if "KEY_W" in game_input:
        HoldKey(0x11)
        time.sleep(6)
        ReleaseKey(0x11)
    elif "KEY_A" in game_input:
        HoldKey(0x1E)
        time.sleep(.5)
        ReleaseKey(0x1E)

def do_second_input(game_input):
    if "KEY_S" in game_input:
        HoldKey(0x1F)
        time.sleep(5)
        ReleaseKey(0x1F)
    elif "KEY_D" in game_input:
        HoldKey(0x20)
        time.sleep(.5)
        ReleaseKey(0x20)

def do_input(game_input):
    if "KEY_SPACE" in game_input:
        HoldKey(0x39)
        time.sleep(.1)
        ReleaseKey(0x39)
    elif "KEY_C" in game_input:
        HoldKey(0x2E)
        time.sleep(.05)
        ReleaseKey(0x2E)
    elif "KEY_1" in game_input:
        HoldKey(0x02)
        time.sleep(.05)
        ReleaseKey(0x02)
    elif "KEY_2" in game_input:
        HoldKey(0x03)
        time.sleep(.05)
        ReleaseKey(0x03)
    elif "KEY_3" in game_input:
        HoldKey(0x04)
        time.sleep(.05)
        ReleaseKey(0x04)

def aim():
    W, H = None, None
    img = sct.grab(origbox)
    # convert image to numpy array
    im = np.array(img)
    frame = cv2.resize(im, (150, 150))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    if W is None or H is None:
        (H, W) = frame.shape[: 2]
    frame = cv2.UMat(frame)
    blob = cv2.dnn.blobFromImage(frame, 1 / 260, (150, 150),
                                 swapRB=False, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        # loop over each of the3 detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]

            # classID = np.argmax(scores)
            # confidence = scores[classID]
            classID = 0  # person = 0
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > CONFIDENCE:
                box = detection[0: 4] * np.array(
                    [W, H, W,
                     H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)
    if len(idxs) > 0:

        # Find best player match
        bestMatch = confidences[np.argmax(confidences)]
        skipRound = False

        # Check if the mouse is already on a target
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            mouseX, mouseY = (
            origbox[0] + x + w / 2, origbox[1] + y + h / 8)
            currentMouseX, currentMouseY = pyautogui.position()

            # Detect closeness to target based on W and H of target
            if abs(mouseX - currentMouseX) < w * 2 and abs(mouseY - currentMouseY) < h * 2:
                skipRound = True

                cv2.circle(frame, (int(x + w / 2), int(y + h / 8)),
                           5, (0, 0, 255), -1)

                if abs(mouseX - currentMouseX) > w * 0.5 or abs(mouseY - currentMouseY) > h * 0.5:
                    set_pos(mouseX, mouseY)

                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 255, 0), 2)
                    text = "TARGET ADJUST {}%".format(
                        int(confidences[i] * 100))
                    cv2.putText(frame, text, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (255, 0, 0), 2)
                    text = "TARGET LOCK {}%".format(
                        int(confidences[i] * 100))
                    cv2.putText(frame, text, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # loop over the indexes we are keeping
            if not skipRound:
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    # draw a bounding box rectangle and label on the frame
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(frame, (x, y),
                                  (x + w, y + h), (0, 0, 255) if bestMatch == confidences[i] else color, 2)

                    text = "TARGET? {}%".format(int(confidences[i] * 100))
                    cv2.putText(frame, text, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if bestMatch == confidences[i]:
                        mouseX = origbox[0] + x + w / 2
                        mouseY = origbox[1] + y + h / 8
                        set_pos(mouseX, mouseY)
                        random_choice = random.randint(1, 2)
                        if random_choice == 1:
                            HoldKey(0x0F)
                            time.sleep(.01)
                            ReleaseKey(0x0F)
                            time.sleep(.1)
                            left_click()
                            HoldKey(0x21)
                            time.sleep(.01)
                            ReleaseKey(0x21)
                            left_click()
                            HoldKey(0x04)
                            time.sleep(.01)
                            ReleaseKey(0x04)
                            left_click()
                        else:
                            HoldKey(0x04)
                            time.sleep(.01)
                            ReleaseKey(0x04)
                            left_click()
                            time.sleep(.15)
                            left_click()
                            time.sleep(.15)
                            left_click()
                            time.sleep(.15)
                            left_click()
                            time.sleep(.15)
                            left_click()
                            time.sleep(.15)
                            left_click()
def run_input(game_input):
    _thread.start_new_thread(aim , ())
    _thread.start_new_thread(do_input, (game_input,))
    _thread.start_new_thread(do_second_input, (game_input,))
    _thread.start_new_thread(do_third_input, (game_input,))
