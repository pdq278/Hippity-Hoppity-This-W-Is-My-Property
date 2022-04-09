# Instructions: Start up Geometry Dash and ensure your windowed resolution is 840x840.
# Make sure that you have autorespawn enabled so if you die you will respawn (and not show the death screen, which we don't want to record).
# When the level begins, run this code.
# Controls: Up key to jump (you must press this so it records it), P to toggle pausing the recording, and Escape to quit and save the recording.
# If the recording somehow gets some garbage (like the death screen), just escape and delete the recording after it saves so we don't mix it with good data.
# When you finish a recording, try to keep the recordings organized in folders named after the level you're recording.
#################################################### IMPORTS ####################################################
from PIL import Image, ImageFilter, ImageGrab
import cv2
import numpy as np
import time
import mss
from pynput.keyboard import Key, Listener, KeyCode, Controller
import keyboard
import os
#################################################### IMPORTS ####################################################


#################################################### GLOBALS ####################################################
def pressSpace():
    print("Pressing space")
    keyboard = Controller() # Create a keyboard.
    keyboard.press(Key.space) # Press the space bar.
    keyboard.release(Key.space) # Release the space bar.

def pressUp():
    print("Pressing up")
    keyboard = Controller() # Create a keyboard.
    keyboard.press(Key.up) # Press the space bar.
    keyboard.release(Key.up) # Release the space bar.

# Captures the screen and returns the image.
def grabScreen():
    ## Grab the screen. ##
    img = None
    with mss.mss() as sct:
        monitor = {"top": 300, "left": 845, "width": 435, "height": 480}
        img = np.array(sct.grab(monitor))
    img = Image.fromarray(img)
    img = img.convert('L')  # Convert to grayscale
    #img = img.point(lambda p: 0 if p < 125  else 255, '1')  # Convert to black and white
    img = img.resize((100, 100))
    ## Grab the screen. ##
    return img

def showInWindow(img):
    frame = np.array(img, dtype=np.float32)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)


def checkIfUpKeyPressed():
    if keyboard.is_pressed('up'):
        return True

def checkIfEscapeKeyPressed():
    if keyboard.is_pressed('esc'):
        return True

def checkIfPPressed():
    if keyboard.is_pressed('p'):
        return True

#################################################### GLOBALS ####################################################


#################################################### MAIN ####################################################
imagesArray = []
actionsArray = []
paused = False
informed = False

while True:

    if (checkIfPPressed()):
        paused = not paused
        time.sleep(0.1)  # Sleep for 0.1 seconds to prevent spamming.

        if (not informed):
            print("Paused: " + str(paused))
            informed = True
        if (not paused):
            informed = False

    if (checkIfUpKeyPressed() and not paused):
        img = grabScreen()
        showInWindow(img)
        img = np.array(img, dtype=np.int32)
        imagesArray.append(img)
        actionsArray.append(1) # 1 for jump
    elif (not paused):
        img = grabScreen()
        showInWindow(img)
        img = np.array(img, dtype=np.int32)
        imagesArray.append(img)
        actionsArray.append(0) # 0 for no jump





    if (checkIfEscapeKeyPressed()):
        break



#################################################### MAIN ####################################################


#################################################### SAVE ####################################################

imagesArray = np.array(imagesArray)
print("Images Array Shape: " + str(imagesArray.shape))
actionsArray = np.array(actionsArray)
print("Actions Array Shape: " + str(actionsArray.shape))

# Check if the file "GeometryDashRecording1.npy" exists.
# If it does, we replace the 1 with 2, 3, 4, etc. until it doesn't exist.
# If it doesn't exist, we create it by saving the array as a numpy array.
i = 1
if (os.path.isfile("GeometryDashRecording" + str(i) + ".npy")):
    i += 1
    while (os.path.isfile("GeometryDashRecording" + str(i) + ".npy")):
        i += 1
    np.save("GeometryDashRecording" + str(i), imagesArray)
    np.save("GeometryDashRecordingActions" + str(i), actionsArray)
else:
    np.save("GeometryDashRecording" + str(i), imagesArray)
    np.save("GeometryDashRecordingActions" + str(i), actionsArray)

#################################################### SAVE ####################################################