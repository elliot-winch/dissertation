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

import json

#NB only works on Windows
import winsound

from architecture import initialize_model, get_architecture_data
from progress_bar import log_progress_bar
import handle_dataloader
import learning_rate_scheduler

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

    epoch_finished_sound_data = [440, 400]
    training_finished_sound_data = [880, 3000]

    def __init__(self, config):
        super(NeuralNetwork, self).__init__()
        self.config = config
        self.output = NeuralNetworkOutput()
        self.output.config = config
        self.output.time = time.ctime()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.architecture_data = None
        self.cancel = False
        self.finished_training = False
        self.on_epoch_finished = []

    def get_num_classes(self):
        return len(self.config.classes)

    def train_from_config(self):

        #Experiment: setting num threads to 1
        torch.set_num_threads(1)

        #Set the seed
        torch.manual_seed(self.config.seed)

        self.init_model()

        #Get image transform
        image_transform = handle_dataloader.default_image_transform(self.architecture_data.image_size)

        class_balance = None
        if hasattr(self.config, 'class_balance'):
            class_balance = self.config.class_balance

        #Data loaders from config
        train_dataloader = handle_dataloader.create_dataloader(
            path = self.config.sorted_data_dir + '/train',
            image_transform = image_transform,
            batch_size = self.config.batch_size,
            class_balance = class_balance
        )

        val_dataloader = handle_dataloader.create_dataloader(
            path = self.config.sorted_data_dir + '/val',
            image_transform = image_transform,
            batch_size = self.config.batch_size
        )

        self.train(train_dataloader, val_dataloader)

    def train(self, train_dataloader, val_dataloader):

        self.init_model()

        #to send the model for training on either cuda or cpu
        self.model = self.model.to(device=self.device)

        weight_tensor = torch.tensor(self.config.loss_function_class_weights, dtype=torch.float, device=self.device)
        criterion = nn.CrossEntropyLoss(weight_tensor)

        parameters_to_update = [param for param in self.model.parameters() if param.requires_grad]
        optimizer = optim.SGD(parameters_to_update, lr=self.config.learning_rate, momentum=self.config.momentum)
        scheduler = learning_rate_scheduler.get_scheduler(self.config, optimizer)

        self.output.epochs = []

        #Run Epochs
        total_time = 0
        for epoch_num in range(self.config.epochs):
            print("\nEpoch " + str(epoch_num)) #Logging

            epoch_output = self.epoch(train_dataloader, val_dataloader, criterion, optimizer, scheduler)
            total_time = total_time + epoch_output.time_taken
            self.output.epochs.append(epoch_output)

            #TODO: might be worth writing an Event class
            [f(self) for f in self.on_epoch_finished]

            if self.cancel:
                break
            else:
                winsound.Beep(self.epoch_finished_sound_data[0], self.epoch_finished_sound_data[1])

        print("\nTotal time (minutes): " + str(total_time / 60))

        self.save_model()

        #Alert training is complete
        if self.cancel is False:
            winsound.Beep(self.training_finished_sound_data[0], self.training_finished_sound_data[1])

    def save_model(self):
        torch.save(self.model.state_dict(), self.config.model_path)
        self.finished_training = True

    def epoch(self, train_dataloader, val_dataloader, criterion, optimizer, scheduler):

        start_time = time.time()

        print("Training...")
        log_progress_bar(0)
        loss_sum = 0.0
        train_num_batches = float(len(train_dataloader))
        for i, data in enumerate(train_dataloader, 0):
            loss_sum += self.epoch_train(data, criterion, optimizer).item()
            log_progress_bar(i / train_num_batches)
        log_progress_bar(1)

        print("\nValidating...")
        log_progress_bar(0)
        val_loss_sum = 0.0
        val_num_batches = float(len(val_dataloader))
        for i, data in enumerate(val_dataloader, 0):
            val_loss_sum += self.epoch_val(data, criterion).item()
            log_progress_bar(i / val_num_batches)
        log_progress_bar(1)

        loss = loss_sum / train_num_batches
        validation_loss = val_loss_sum / val_num_batches

        learning_rate_scheduler.step_scheduler(scheduler,self.config, validation_loss)

        epoch_output = EpochOutput()
        epoch_output.loss = loss
        epoch_output.validation_loss = validation_loss
        epoch_output.learning_rate = optimizer.param_groups[0]['lr']
        epoch_output.confusion_matrix = self.confusion_matrix(val_dataloader)

        time_taken = time.time() - start_time
        print("\nEpoch time (minutes): " + str(time_taken / 60))
        epoch_output.time_taken = time_taken

        return epoch_output

    def epoch_val(self, data, criterion):

        self.model.eval()

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(self.device), labels.to(self.device) # Send data to C/GPU

        outputs = self.model(inputs)
        return criterion(outputs, labels)

    def epoch_train(self, data, criterion, optimizer):

        self.model.train()

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(self.device), labels.to(self.device) # Send data to C/GPU

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        if self.architecture_data.use_aux:
            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
            outputs, aux_outputs = self.model(inputs)
            loss1 = criterion(outputs, labels)
            loss2 = criterion(aux_outputs, labels)
            loss = loss1 + 0.4*loss2
        else:
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        return loss

    def test_from_config(self):

        self.init_model(load=True);

        image_transform = handle_dataloader.default_image_transform(self.architecture_data.image_size)
        test_dataloader = handle_dataloader.create_dataloader(
            path = self.config.sorted_data_dir + '/test',
            image_transform = image_transform,
            batch_size = self.config.batch_size
        )

        self.output.confusion_matrix = self.confusion_matrix(test_dataloader)

    def confusion_matrix(self, dataloader):
        pred = []
        true = []

        for inputs, labels in dataloader:
            # Send data to C/GPU
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # Feed Network
            output = self.model(inputs)
            #Converts prediction scores into class predictions
            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            # Save Prediction
            pred.extend(output)
            # Save labels
            labels = labels.data.cpu().numpy()
            true.extend(labels)

        num_classes = self.get_num_classes()
        confusion_matrix = [ [0]*num_classes for i in range(num_classes)]

        for i in range(num_classes):
            confusion_matrix[true[i]][pred[i]] += 1

        return confusion_matrix

    def init_model(self, load=False):
        #Loads model from file if it's not already in memory
        if self.model is None:
            self.model, self.architecture_data = initialize_model(self.config.model_name, self.get_num_classes(), self.config.use_transfer_learning)

            if load:
                self.model.load_state_dict(torch.load(self.config.model_path))
