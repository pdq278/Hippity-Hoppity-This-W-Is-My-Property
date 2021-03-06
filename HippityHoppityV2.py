import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader as dataloader
import numpy as np
import matplotlib.pyplot as plt

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
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(500, 750)
        self.relu6 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.25)
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
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu6(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = self.relu7(out)
        out = self.fc4(out)
        out = self.sigmoid(out)

        return out



    def train(self, x, y, learningRate, epochs, batchSize):
        # Loss and optimizer
        # Use binary cross entropy loss
        loss = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=learningRate)
        # Train the model
        avgLoss = 0
        prevAvgLoss = 999
        lossList = []
        avgLossList = []
        for epoch in range(epochs):
            lossValue = None
            for i in range(0, len(x), batchSize):
                # Get the inputs
                inputs = x[i:i + batchSize]
                # Get the targets
                targets = y[i:i + batchSize]
                # Forward pass
                outputs = self(inputs.unsqueeze(1))
                # Compute the loss
                lossValue = loss(outputs, targets)
                lossList.append(lossValue.item())
                # Backward pass after each batch
                optimizer.zero_grad()
                lossValue.backward()
                # Update the weights
                optimizer.step()


            length = len(lossList)
            avgLoss = sum(lossList) / length
            avgLossList.append(avgLoss)
            if (avgLoss > prevAvgLoss): # If the loss is increasing, we want to decrease the learning rate
                learningRate = learningRate * 0.5
                print("Learning rate decreased to: " + str(learningRate))
                optimizer = optim.Adam(self.parameters(), lr=learningRate)
            prevAvgLoss = avgLoss
            print("Epoch: ", epoch, "Loss: ", avgLoss)
            if (epoch % 4 == 0 and epoch != 0):
                #self.plotLoss(lossList)
                self.plotLoss(avgLossList)
        return self, lossList

    def predict(self, x):
        return self(x)

    # Save the model
    def save(self, path):
        torch.save(self.state_dict(), path)

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

    def plotLoss(self, lossList):
        plt.plot(lossList)
        i = 1
        while (os.path.exists("loss" + str(i) + ".png")):
            i += 1
        plt.savefig("loss" + str(i) + ".png")
        #plt.clf() # Clear the figure
        print("Saved loss plot")



# Epochs
epochs = 3
# Batch size
batch_size = 1
# Learning rate, we want to use a very small learning rate
learning_rate = 1e-8 # 1e-6


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on device: ", device)


model = NeuralNetwork()
m = "HippityHoppityMAIN_1400.pt" #
if (os.path.isfile(m)):
    print("Loading model: " + m)
    model.load(m)

if (torch.cuda.is_available()):
    model.to(device)



def trainOnData(model, dataXPath, dataYPath, learning_rate, epochs, batch_size):
    ########################################################################################################################
    # Load the numpy array called "GeometryDashRecording1.npy" as our dataX. Don't forget to unpickle it!
    # Load the numpy array called "GeometryDashRecordingActions1.npy" as our dataY. Don't forget to unpickle it!
    #dataX = np.load(r"L1\GeometryDashRecording1.npy", allow_pickle=True)
    dataX = np.load(dataXPath, allow_pickle=True)
    #dataY = np.load(r"L1\GeometryDashRecordingActions1.npy", allow_pickle=True)
    dataY = np.load(dataYPath, allow_pickle=True)
    # Reshape dataX to be [n_samples, 1, 100, 100]
    dataX = dataX.reshape(-1, 100, 100).astype(np.float32)
    # Reshape dataY to be [n_samples, 1]
    dataY = dataY.reshape(-1, 1).astype(np.float32)
    print("DataX shape: ", dataX.shape)
    print("DataY shape: ", dataY.shape)
    print("An example of dataY: ", dataY[1])

    # To tensor
    dataX = model.toTensor(dataX)
    dataY = model.toTensor(dataY)

    # If CUDA is available, use it!
    if torch.cuda.is_available():
        dataX = dataX.to(device)
        dataY = dataY.to(device)

    # Train the model
    _, lossList = model.train(dataX, dataY, learning_rate, epochs, batch_size)
    ########################################################################################################################

def saveModel(model):
    # Save the model
    name = "HippityHoppityMAIN_"
    i = 1
    if (os.path.isfile(name + str(i) + ".pt")):
        while (os.path.isfile(name + str(i) + ".pt")):
            i += 1
        model.save(name + str(i) + ".pt")
    else:
        model.save(name + str(i) + ".pt")



# We train on all the data.
#levels = ["New Stereo Madness", "New Back On Track"]
#recordings = [[1,2,3,4,5,6], [1,2,3,4,5,6]]
levels = ["New Stereo Madness", "New Back On Track", "New Polargeist", "New Cant Let Go", "New Base After Base", "New Jumper"]
recordings = [[1,2,3,4,5,6,7], [1,2,3,4,5,6,7], [1,2,3,4,5], [1,2,3,4,5,6,7], [1,2,3,4,5,6,7], [1,2,3,4,5]]
sessions = 20

for i in range(sessions):
    for recording in range(1, 8):
        for levelIndex in range(len(levels)):
            if (recording in recordings[levelIndex]):
                # index is the index of where the recording number is in the recordings list.
                index = recordings[levelIndex].index(recording)
                print("Training on level: ", levels[levelIndex] + " Recording: ", recording)
                trainOnData(model, levels[levelIndex] + "\\" + "GeometryDashRecording" + str(recordings[levelIndex][index]) + ".npy", levels[levelIndex] + "\\" + "GeometryDashRecordingActions" + str(recordings[levelIndex][index]) + ".npy", learning_rate, epochs, batch_size)
                saveModel(model)
