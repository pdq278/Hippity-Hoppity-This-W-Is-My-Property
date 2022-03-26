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


## Other variables ##
fps = 240 # I just set this to 240 because it's a high number. I don't know if that's a good idea.
prev = 0
## Other variables ##


## Our agent's variables ##
currentTime = time.time() # Set the time to the current time. This is to be used for the time reward.
global jumps
jumps = 0 # This is the number of jumps we have made.
## Our agent's variables ##



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
    if (abs(whitePixels - 287059)/287059) < 0.005: # From a test, I found that 287059 is the average of the white pixels in the image so
        amDead = True
        return True # Return true since we are on the replay button (death screen).
    else:
        return False # Return false since we are not on the replay button (death screen).

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
    time_elapsed = time.time() - prev
    img = None
    with mss.mss() as sct:
        monitor = {"top": 30, "left": 270, "width": 1370, "height": 1050}
        img = np.array(sct.grab(monitor))
    img = Image.fromarray(img)
    img = img.resize((1000, 1000))
    # Remove the alpha channel.
    # Pixelate the image
    # img = img.resize((int(img.width/12), int(img.height/12)), resample=Image.BILINEAR).resize(SCREEN_SIZE,Image.NEAREST)
    img = img.filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 7, -1, -1, -1, -1), 1, 0))  # Edge detection
    img = img.convert("L")  # Convert to grayscale
    ## Grab the screen. ##
    return img

# Checks if the replay button is on the screen, if so it presses space (restarting the game).
def checkAndReset(currentTime):
    ## Death reset stuff. ##
    dead = checkIfDead() # Check if we are on the replay button.
    if (dead):  # If we are dead, press space to restart.
        pressSpace()  # Press space to restart.
    return dead
    ## Death reset stuff. ##


# Shows the image in a window. If you place this in a while loop with grabScreen() it will repeatedly update the screen.
def showInWindow(img):
    frame = np.array(img)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

# This function is needed to convert our image into a tensor so we can feed it into the neural network.
def imgToTensor(img):
    imgTensor = img.transpose(2, 0, 1)
    imgTensor = imgTensor.astype(np.float32)
    imgTensor = torch.from_numpy(imgTensor)
    if (torch.cuda.is_available()):
        imgTensor = imgTensor.cuda()
    return imgTensor

def doAction(actionIndex):
    print("Action Index: " + str(actionIndex))
    j = jumps
    if actionIndex[0] == 1:
        pressUp()
        j += 1
        return 1
    else:
        print("Doing nothing")
        return 0



def getTimeReward(startTime):
    # timeReward is equal to the current time - the startTime (the time when we last reset).
    timeReward = time.time() - startTime
    return timeReward


def frameToState(action, currentTime):
    doAction(action)
    reward = 0
    terminal = False
    # We wait for a frame to pass before we grab the screen.
    img = np.array(grabScreen()).reshape(1000, 1000, 1)
    img = imgToTensor(img)

    terminal = checkAndReset(currentTime)
    if (terminal):
        currentTime    = time.time()
        reward = -10
    else:
        reward = getTimeReward(currentTime)*10 + jumps
    return img, reward, terminal



class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # Hyperparameters chosen based on DeepMind's paper.
        self.number_of_actions = 2 # Number of actions.
        self.gamma = 0.99 # Discount factor. Used to calculate the discounted reward, which is the reward that the agent will get for the current state.
        self.final_epsilon = 0.0001 # Final value of epsilon.
        self.initial_epsilon = 0.1 # This determines how often we will explore.
        self.number_of_iterations = 2000000 # Number of times we will train the network.
        self.replay_memory_size = 10000 # Number of experiences we will store in the replay memory.
        self.minibatch_size = 10 # Number of experiences we will sample from the replay memory.




        self.conv1 = nn.Conv2d(4, 32, 4, 1) # Input channels, output channels, kernel size, stride
        self.relu1 = nn.ReLU(inplace=True) # Inplace is true so we don't create a new tensor.
        self.conv2 = nn.Conv2d(32, 64, 6, 2 )
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, 8,  8)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(64, 64, 10,  16)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(1024, 2024)
        self.relu5 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(2024, self.number_of_actions)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = out.view(out.size(0), -1)
        out = self.fc4(out)
        out = self.relu5(out)
        out = self.fc5(out)

        return out

model = NeuralNetwork()
optimizer = optim.Adam(model.parameters(), lr=1e-6)
criterion = nn.MSELoss()

replay_memory = []

action = torch.zeros([model.number_of_actions], dtype=torch.float32)
action[0] = 1
imgData, reward, terminal = frameToState(action, currentTime)
state = torch.cat((imgData, imgData, imgData, imgData)).unsqueeze(0)

epsilon = model.initial_epsilon
iteration = 0

epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)


epsilon = model.initial_epsilon


start = time.time()

while iteration < model.number_of_iterations:
    print("Terminal Status At Start: " + str(terminal))
    # get output from the neural network
    output = model(state)[0]

    # initialize action
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    if torch.cuda.is_available():  # put on GPU if CUDA is available
        action = action.cuda()

    # epsilon greedy exploration
    random_action = random.random() <= epsilon
    if random_action:
        print("Performed random action!")
    action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                    if random_action
                    else torch.argmax(output)][0]

    if torch.cuda.is_available():  # put on GPU if CUDA is available
        action_index = action_index.cuda()

    action[action_index] = 1


    # get next state and reward
    image_data_1, reward, terminal = frameToState(action, currentTime)
    state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

    action = action.unsqueeze(0)
    reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)


    # save transition to replay memory
    replay_memory.append((state, action, reward, state_1, terminal))

    # if replay memory is full, remove the oldest transition
    if len(replay_memory) > model.replay_memory_size:
        replay_memory.pop(0)

        # epsilon annealing
        epsilon = epsilon_decrements[iteration]

        # sample random minibatch
        minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

        # unpack minibatch
        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_1_batch = state_1_batch.cuda()

        # get output for the next state
        output_1_batch = model(state_1_batch)

        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + model.gamma * torch.max(output_1_batch[i])
                                  for i in range(len(minibatch))))

        # extract Q-value
        q_value = torch.sum(model(state_batch) * action_batch, dim=1)

        # PyTorch accumulates gradients by default, so they need to be reset in each pass
        optimizer.zero_grad()

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()

        # calculate loss
        loss = criterion(q_value, y_batch)

        # do backward pass
        loss.backward()
        optimizer.step()

    #print("TERMINAL VALUE: ", terminal)
    if (terminal):
        terminal = False

        #print("Old Current Time: " + str(currentTime))
        currentTime = time.time() # reset time
        jumps = 0 # reset jumps
        #print("New Current Time: " + str(currentTime))



    # set state to be state_1
    state = state_1
    iteration += 1

    if iteration % 25000 == 0:
        torch.save(model, "current_model_" + str(iteration) + ".pth")

    print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
          action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
          np.max(output.cpu().detach().numpy()))







