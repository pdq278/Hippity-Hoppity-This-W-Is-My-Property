import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFilter, ImageGrab
import cv2
import numpy as np
import time
import mss
from pynput.keyboard import Key, Listener, KeyCode, Controller
import keyboard
import os
import torch.utils.data.dataloader as dataloader
import numpy as np
import matplotlib.pyplot as plt

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
    # Blur the img slightly
    #img = img.point(lambda p: 0 if p < 125  else 255, '1')  # Convert to black and white
    #img = img.filter(ImageFilter.BLUR)
    img = img.resize((100, 100))
    ## Grab the screen. ##
    return img

def showInWindow(img):
    frame = np.array(img, dtype=np.float32)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

def checkIfEscapeKeyPressed():
    if keyboard.is_pressed('esc'):
        return True


#################################################### GLOBALS ####################################################

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Convolutional layers
        self.layer1 = nn.Conv2d(1, 32, kernel_size=5, stride=3, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=2)
        self.relu2 = nn.ReLU()
        self.layer3= nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=2)
        self.relu3 = nn.ReLU()
        self.layer4 = nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=2)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers
        self.fc1 = nn.Linear(16, 500)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(500, 750)
        self.relu6 = nn.ReLU()
        self.fc3 = nn.Linear(750, 200) # We want our output between 0 and 1, so we use a sigmoid activation function
        self.relu7 = nn.ReLU()
        self.fc4 = nn.Linear(200, 1)
        self.sigmoid = nn.Sigmoid()



    def forward(self, input):
        # Convolutional layers
        out = self.layer1(input)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.layer2(out)
        out = self.relu2(out)
        out = self.layer3(out)
        out = self.relu3(out)
        out = self.layer4(out)
        out = self.relu4(out)
        out = self.pool2(out)
        # Fully connected layers
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu5(out)
        out = self.fc2(out)
        out = self.relu6(out)
        out = self.fc3(out)
        out = self.relu7(out)
        out = self.fc4(out)
        out = self.sigmoid(out)

        return out

    def predict(self, x):
        return self(x)

    # Load the model
    def load(self, path):
        self.load_state_dict(torch.load(path))

    # Test the model
    def test(self, x, y):
        output = self(x)
        loss = nn.MSELoss(output, y)
        return loss.item()

    def toTensor(self, x):
        return torch.tensor(x)



model = NeuralNetwork()
model.load("HippityHoppityBIG_4.pt")

while not checkIfEscapeKeyPressed():
    img = grabScreen()
    #showInWindow(img)

    img = np.array(img, dtype=np.float32)
    img = img.reshape(1, 1, 100, 100)
    img = model.toTensor(img)
    prediction = model.predict(img).detach()[0].item()
    # Round the prediction to the nearest integer
    print(prediction)
    prediction = 1 if prediction >= 0.75 else 0
    if (prediction == 1):
        pressUp()
    else:
        #print("Doing nothing.")
        pass
