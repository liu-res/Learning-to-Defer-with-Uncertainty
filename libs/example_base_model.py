import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
from sklearn.metrics import classification_report
import os


class base_model_class(nn.Module):

    def __init__(self):
        super(base_model_class, self).__init__()
        self.input_size = 784
        self.hidden_sizes = [128, 64]
        self.output_size = 10
        self.l1 = nn.Linear(self.input_size, self.hidden_sizes[0])
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])
        self.l3 = nn.Linear(self.hidden_sizes[1], self.output_size)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        return F.softmax(x, dim=1)


def trainer(model):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    path_to_trainset = '../demo_data/'
    if not os.path.exists(path_to_trainset):
        os.makedirs(path_to_trainset)

    dataset = datasets.MNIST(path_to_trainset, download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    device = 'cpu'
    epochs = 15
    loss_function = nn.NLLLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    time0 = time()
    epochs = epochs
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
            # Training pass
            optimizer.zero_grad()
            output = model(images)
            loss = loss_function(output, labels)
            # This is where the model learns by backpropagating
            loss.backward()
            # And optimizes its weights here
            optimizer.step()
            running_loss += loss.item()
        print("Epoch {} - Training loss: {}".format(e, running_loss / len(trainloader)))
    print("\nTraining Time (in minutes) =", (time() - time0) / 60)
    return model


def validator(model, train=False, shuffle=False):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    path_to_testset = '../demo_data/'
    if not os.path.exists(path_to_testset):
        os.makedirs(path_to_testset)

    dataset = datasets.MNIST(path_to_testset, download=True, train=train, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=shuffle)

    with torch.set_grad_enabled(False):
        label = []
        pred = []
        prob_NP = np.empty((0, model.output_size)).astype(float)  # 2-D array
        for images,labels in dataloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
            out_mini = model(images)
            label.extend(labels.tolist())
            pred.extend(torch.max(out_mini.cpu().data, 1)[1].numpy().astype(int).tolist())
            prob_NP = np.concatenate((prob_NP, out_mini.cpu().data.numpy().astype(float)), axis=0)
            del images
            del out_mini
        print('\nClassification Report\n')
        print(classification_report(label, pred, target_names=['cls{}'.format(i) for i in range(model.output_size)]))
    return label, pred, prob_NP


