import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time
import torch
import torchvision
from torch import nn, optim
from torchvision import datasets, transforms
import torch.nn.functional as F

# MNIST DATA LOAD
batch_size_train = 64
batch_size_test = 64
trans = transforms.Compose([torchvision.transforms.ToTensor(), 
                            transforms.Normalize((0.5,), (0.5,)),
                           ])
root = '/files/'

# Set of All Data
trainset = datasets.MNIST(root=root, train=True, transform=trans, download=True)
testset =  datasets.MNIST(root=root, train=False, transform=trans, download=True)

# Reduce Train Data to digits 1 & 0
idx = (trainset.targets == 0) | (trainset.targets == 1)
trainset.data, trainset.targets = trainset.data[idx], trainset.targets[idx]

# Reduce Test Data to digits 1 & 0
idx_test = (testset.targets == 0) | (testset.targets == 1)
testset.data, testset.targets = testset.data[idx_test], testset.targets[idx_test]


# Train Data
trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                           batch_size=batch_size_train,
                                           shuffle=True)

# Test Data
testloader = torch.utils.data.DataLoader(dataset=testset,
                                          batch_size=batch_size_test,
                                          shuffle=True)

print (f'==>>> total trainning batch number: {len(trainloader)}')
print (f'==>>> total testing batch number: {len(testloader)}')

train_iter = iter(trainloader)
images, labels = train_iter.next()

test_iter = iter(testloader)
testimages, testlabels = test_iter.next()

# LOOK AT IMAGES, DIMS, and LABELS
# print(f'{images.shape[0]} images in one mini-batch of dim: {images.shape[1:4]}')
# print(f'{labels.shape} labels in one mini-batch')

# # Plot
# fig = plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(images[0].numpy().squeeze(), cmap='gray_r');
# plt.title(f"Corresponding Label: {labels[0]}")

# plt.subplot(1,2,2)
# plt.imshow(testimages[0].numpy().squeeze(), cmap='gray_r');
# plt.title(f"Corresponding Label: {testlabels[0]}")

# Define Train Loop
def train(model, device, trainloader, loss_fn, optimizer, epochs):
    model.train()
    running_loss = 0.0
    loss_values = []
    for batch_idx, (data, target) in enumerate(trainloader):
        data = data.view(data.shape[0], -1)  # Flatten 28x28 image
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()                             # 1. Zero out gradient
        output = model(data)                              # 2. Predict
        loss = loss_fn(output, target)                    # 3. Loss calculation
        loss.backward()                                   # 4. Backprop learning
        optimizer.step()                                  # 5. Step down gradient
        running_loss += loss.item() * data.size(0)        # 6. Accumulate loss
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} -- [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, 
                batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), 
                loss.item())
                 )        
    return running_loss
  
# Define Test Loop
def test(model, device, testloader):
    model.eval()
    running_test_loss = 0.0
    running_test_acc = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data = data.view(data.shape[0], -1)  # Flatten 28x28 image
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss_fn = nn.NLLLoss(reduction='sum')
            running_test_loss += loss_fn(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()  # not sure if it's this or the line below
            correct += pred.eq(target.data.view_as(pred)).sum()
    
    running_test_loss /= len(testloader.dataset)
    running_test_acc = 100. * correct / len(testloader.dataset)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        running_test_loss, 
        correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset))
         )
    return running_test_loss, running_test_acc
  
# Model (capable of cuda or cpu)
device = "cpu" # "cuda" or "cpu"
input_size = int(28*28)  # which is 784
hidden_size = 64
output_size = 2

# data shape must match input layer
model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.Sigmoid(),
    nn.Linear(hidden_size, hidden_size),  # hidden=64
    nn.Sigmoid(),
    nn.Linear(hidden_size, output_size),
    nn.LogSoftmax(dim=1)                  # output_classes=1 (wither 1 or 0)
).to(device)

# Params
epochs = 10
learning_rate = 0.01
momentum = 0.5
log_interval = 10

# Define Loss & Optim
loss_fn = nn.NLLLoss()     # negative log-likelihood (cross entropy loss)
optimizer = optim.SGD(model.parameters(), 
                      lr=learning_rate,
                      momentum=momentum)
  
if __name__ == "__main__":
  print("Prepring Training & Testing:")
  print("Loss Function: Negative Log-likelihood")
  print("Optimizer: SGD")
  print(f"Learn Rate: {learning_rate}")
  print(f"Train Batch size: {batch_size_train}")
  print(f"Num Epochs: {epochs}\n")

  train_loss = []
  test_loss = []
  test_accuracy = []
  time0 = time()

  # Test (untrained weights)
  running_test_loss, running_test_acc = test(model, device, testloader)  # run manualy without any learnig
  test_loss.append(running_test_loss)
  test_accuracy.append(running_test_acc)

  for epoch in range(epochs):

      # train
      running_loss = train(model, device, trainloader, loss_fn, optimizer, epoch)
      train_loss.append(running_loss / len(trainset))

      # test
      running_test_loss, running_test_acc = test(model, device, testloader)
      test_loss.append(running_test_loss)
      test_accuracy.append(running_test_acc)

  print(f'\nTraining Time (in minutes) = {((time() - time0)/60):.2f}')

  # Plot Results
  fig = plt.figure()

  plt.subplot(1,2,1)
  plt.plot(train_loss)
  plt.xlabel("Epochs")
  plt.ylabel("Training Loss")
  plt.title("Train Loss vs. Epoch")

  plt.subplot(1,2,2)
  plt.plot(test_accuracy)
  plt.xlabel("Epochs")
  plt.ylabel("Test Accuracy (%)")
  plt.title("Test Accuracy vs. Epoch")
