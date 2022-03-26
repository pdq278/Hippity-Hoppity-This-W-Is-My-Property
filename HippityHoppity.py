## Regular Imports ##
from PIL import Image, ImageFilter, ImageGrab
import cv2
import numpy as np
import time
import mss
from pynput.keyboard import Key, Controller
import random
from collections import namedtuple, deque
from itertools import count
## Custom Imports ##

## PyTorch Imports ##
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
## PyTorch Imports ##


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # sets device to cuda if available, otherwise we'll just use the cpu.


# This function checks if the pixels are a certain color in a section of the image. If so, it means we died.
# The monitor dimensions need to be set up such that it is centered on the location of the screen where the replay button will be when you died.
def checkIfDead():
    with mss.mss() as sct:
        monitor = {"top": 850, "left": 620, "width": 100, "height": 100} # This is the location of the replay button.
        img = np.array(sct.grab(monitor)) # Grab the pixels of the screen.
    img = Image.fromarray(img) # Convert the array to an image.
    img = img.convert("L") # Convert to grayscale
    img = img.filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8, -1, -1, -1, -1), 1, 0)) # Edge detection
    # Sum the white pixels in the image.
    whitePixels = np.sum(img) # Sum the white pixels in the image.
    if (abs(whitePixels - 287059)/287059) < 0.006: # From a test, I found that 287059 is the average of the white pixels in the image so
        return True # Return true since we are on the replay button (death screen).
    else:
        return False # Return false since we are not on the replay button (death screen).

def pressSpace():
    print("Pressing space")
    keyboard = Controller() # Create a keyboard.
    keyboard.press(Key.space) # Press the space bar.
    keyboard.release(Key.space) # Release the space bar.

# Captures the screen and returns the image.
def grabScreen():
    ## Grab the screen. ##
    time_elapsed = time.time() - prev
    img = None
    with mss.mss() as sct:
        monitor = {"top": 30, "left": 270, "width": 1370, "height": 1050}
        img = np.array(sct.grab(monitor))
    img = Image.fromarray(img)
    # Pixelate the image
    # img = img.resize((int(img.width/12), int(img.height/12)), resample=Image.BILINEAR).resize(SCREEN_SIZE,Image.NEAREST)
    img = img.filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 7.5, -1, -1, -1, -1), 1, 0))  # Edge detection
    img = img.convert("L")  # Convert to grayscale
    ## Grab the screen. ##
    return img

# Checks if the replay button is on the screen, if so it presses space (restarting the game).
def checkAndReset():
    ## Death reset stuff. ##
    resetDeath = False  # This boolean is included to only reset the death once.
    if (checkIfDead() and not resetDeath):  # If we are dead, press space to restart.
        pressSpace()  # Press space to restart.
        resetDeath = True  # Set the resetDeath boolean to true so we don't press space again. We don't want to jump a bunch of time for no reason.
        timeReward = 0  # Reset the timeReward.
    ## Death reset stuff. ##

# Shows the image in a window. If you place this in a while loop with grabScreen() it will repeatedly update the screen.
def showInWindow(img):
    frame = np.array(img)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

# This function is needed to convert our image into a tensor so we can feed it into the neural network.
def imgToTensor(img):
    img = T.ToTensor()(img) # Convert the image to a tensor.
    img = img.unsqueeze(0)  # Add a dimension to the tensor.
    return img # Return the tensor.

## Other variables ##
fps = 240 # I just set this to 240 because it's a high number. I don't know if that's a good idea.
prev = 0
## Other variables ##


## Our agent's variables ##
timeReward = 0 # Time-based reward for our agent.

## Our agent's variables ##


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # Hyperparameters chosen based on DeepMind's paper.
        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 2000000
        self.replay_memory_size = 10000
        self.minibatch_size = 32

        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(3136, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(512, self.number_of_actions)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)

        return out

model = NeuralNetwork()
optimizer = optim.Adam(model.parameters(), lr=1e-6)
criterion = nn.MSELoss()

replay_memory = []

# Test
while True: # Our main loop.

    checkAndReset()
    img = grabScreen()
    showInWindow(img)
