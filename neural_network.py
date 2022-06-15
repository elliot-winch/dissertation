import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import torch.optim as optim

import torchvision

import numpy as np

import time

from ignite.metrics.confusion_matrix import ConfusionMatrix

import json

#NB only works on Windows
import winsound

from architecture import initialize_model
import arrange_files

class NeuralNetworkOutput(object):
    pass

class EpochOutput(object):
    pass

"""
Wrapped class for training & testing a Neural Network

#caution: each new model is ~0.2GB. Creating lots of models will quickly take up memory.
Ensure model name is the same as existing if you don't want to create a new file
"""
class NeuralNetwork(object):

    config = {}
    epoch_finished_sound_data = [440, 400]
    training_finished_sound_data = [880, 3000]

    device = None

    output = NeuralNetworkOutput()

    def __init__(self, config):
        super(NeuralNetwork, self).__init__()
        self.config = config
        self.output.config = config.__dict__

    def train(self, needs_arrange=True):

        torch.manual_seed(self.config.seed)

        if needs_arrange:
            arrange_files.arrange_files(self.config)

        num_classes = len(self.config.classes)
        model, input_size = initialize_model(self.config.model_name, num_classes, self.config.use_transfer_learning)

        print(model)

        #Data loaders
        image_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            #Numbers specified by PyTorch
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_dataset = datasets.ImageFolder(self.config.sorted_data_dir + '/train', transform=image_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)

        val_dataset = datasets.ImageFolder(self.config.sorted_data_dir + '/val', transform=image_transform)
        val_dataloader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device=self.device) #to send the model for training on either cuda or cpu

        criterion = nn.CrossEntropyLoss() #Adjust this function to use different loss

        parameters_to_update = [param for param in model.parameters() if param.requires_grad]
        optimizer = optim.SGD(parameters_to_update, lr=self.config.learning_rate, momentum=self.config.momentum)

        total_time = 0

        self.output.epochs = []

        #Run Epochs
        for epoch_num in range(self.config.epochs):

            print("\nEpoch " + str(epoch_num)) #Logging

            epoch_output = self.epoch(train_dataloader, val_dataloader, model, criterion, optimizer)
            total_time = total_time + epoch_output.time_taken
            winsound.Beep(self.epoch_finished_sound_data[0], self.epoch_finished_sound_data[1])

            self.output.epochs.append(epoch_output.__dict__)

        print("\nTotal time (minutes): " + str(total_time / 60))

        torch.save(model.state_dict(), self.config.model_path)

        #Alert training is complete
        winsound.Beep(self.training_finished_sound_data[0], self.training_finished_sound_data[1])

    def epoch(self, train_dataloader, val_dataloader, model, criterion, optimizer):

        start_time = time.time()
        num_batches = len(train_dataloader)

        print("Training...")
        self.log_progress_bar(0)
        loss_sum = 0.0
        for i, data in enumerate(train_dataloader, 0):
            loss_sum += self.epoch_train(data, model, criterion, optimizer).item()
            self.log_progress_bar(i / num_batches)
        self.log_progress_bar(1)

        print("\nValidating...")
        self.log_progress_bar(0)
        val_loss_sum = 0.0
        for i, data in enumerate(val_dataloader, 0):
            val_loss_sum += self.epoch_val(data, model, criterion).item()
            self.log_progress_bar(i / num_batches)
        self.log_progress_bar(1)

        loss = loss_sum / len(train_dataloader)
        validation_loss = val_loss_sum / len(val_dataloader)

        time_taken = time.time() - start_time
        print("\nEpoch time (minutes): " + str(time_taken / 60))

        epoch_output = EpochOutput()
        epoch_output.loss = loss
        epoch_output.validation_loss = validation_loss
        epoch_output.time_taken = time_taken

        return epoch_output

    def epoch_val(self, data, model, criterion):

        model.eval()

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(self.device), labels.to(self.device) # Send data to C/GPU

        outputs = model(inputs)
        return criterion(outputs, labels)

    def epoch_train(self, data, model, criterion, optimizer):

        model.train()

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(self.device), labels.to(self.device) # Send data to C/GPU

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        return loss

    def test(self):
        num_classes = len(self.config.classes)
        model, input_size = initialize_model(self.config.model_name, num_classes, self.config.use_transfer_learning)
        model.load_state_dict(torch.load(self.config.model_path))

        test_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor()
        ])

        test_dataset = datasets.ImageFolder(self.config.sorted_data_dir + '/test', transform=test_transform)
        test_dataloader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=True)

        y_pred = []
        y_true = []

        # iterate over test data
        for inputs, labels in test_dataloader:
            output = model(inputs) # Feed Network

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output) # Save Prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels) # Save Truth

        num_classes = len(self.config.classes)
        confusion_matrix = [ [0]*num_classes for i in range(num_classes)]

        for i in range(len(y_true)):
            confusion_matrix[y_true[i]][y_pred[i]] += 1

        self.output.confusion_matrix = confusion_matrix

    def mean_average_precision(self, matrix):
        #matrix should be square
        total = 0
        total_correct = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                total += matrix[i][j]

                if i is j:
                    total_correct += matrix[i][j]

        return total_correct / float(total)

    #logging - perhaps define in a new file
    def log_progress_bar(self, percent=0, width=40):
        left = int(width * percent)
        right = width - left

        tags = "#" * left
        spaces = " " * right
        percents = f"{(percent * 100):.0f}%"

        print("\r[", tags, spaces, "]", percents, sep="", end="", flush=True)
