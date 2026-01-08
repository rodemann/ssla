import argparse
import os
import shutil
import time
import sys
import csv
import json
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from bayesian_torch.layers import Conv2dReparameterization
from bayesian_torch.layers import LinearReparameterization
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import bayesian_torch.models.bayesian.resnet_variational as resnet
import numpy as np


def create_regression_data(glucose_src, insulin_src, split_ratio, history_length = 10, prediction_horizon = 5):

    assert split_ratio > 0.0 and split_ratio <= 1.0

    random.seed(10)

    assert glucose_src.find('csv') > -1
    assert insulin_src.find('csv') > -1

    glucose_matrix = []
    with open(glucose_src) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            trace = []
            for each_entry in row :
                trace.append(float(each_entry))
            glucose_matrix.append(trace)

    insulin_matrix = []
    with open(insulin_src) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            trace = []
            for each_entry in row :
                trace.append(float(each_entry))
            insulin_matrix.append(trace)

    assert len(glucose_matrix) > 0
    assert len(glucose_matrix) == len(insulin_matrix)
    assert len(glucose_matrix[0]) == len(insulin_matrix[0])


    all_data = []
    no_of_traces = len(glucose_matrix)

    for trace_index in range(no_of_traces):
        trace_length = len(glucose_matrix[0])
        assert trace_length == len(insulin_matrix[0])
        for start_time in range(history_length, trace_length - (prediction_horizon), 1):
            end_time = start_time + prediction_horizon

            glucose_slice = glucose_matrix[trace_index][start_time - history_length : start_time]
            insulin_slice = insulin_matrix[trace_index][start_time - history_length : start_time]
            glucose_prediction = glucose_matrix[trace_index][end_time]

            training_pair = ( glucose_slice + insulin_slice, glucose_prediction)
            all_data.append(training_pair)

    random.shuffle(all_data)

    cut_point = int(float(split_ratio) * float(len(all_data)))
    train_data = all_data[: cut_point]
    test_data = all_data[cut_point:]


    return train_data, test_data

def generate_data(data_points = 100):

    xvalues = np.linspace(0, 10, data_points, dtype = np.double).reshape(data_points, 1)
    np.random.shuffle(xvalues)
    yvalues = np.zeros((data_points, 1))

    for idx in range(len(xvalues)):
        yvalues[idx] = [np.power(xvalues[idx][0], 2.0, dtype = np.double)]

    return xvalues, yvalues


class diabetes_loader(datasets.VisionDataset):
    def __init__(self, train = True, test = False):

        super(diabetes_loader, self)
        glucose_src = "./data/Glucose_data_no_meals.csv"
        insulin_src = "./data/Insulin_data_no_meals.csv"
        train_data, test_data = create_regression_data(glucose_src, insulin_src, split_ratio = 0.8)
        if train:
            self.data = train_data
        if test:
            self.data = test_data

        assert len(self.data) > 0

    def __getitem__(self, index):
        input, target = self.data[index][0], self.data[index][1]
        target = [target]
        input = np.asarray(input)
        target = np.asarray(target)

        input = torch.from_numpy(input)
        target = torch.from_numpy(target)

        return input, target

    def __len__(self):
        return len(self.data)


class custom_loader(datasets.VisionDataset):
    def __init__(self, train = True, test = False):

        super(custom_loader, self)

        if train:
            self.data, self.targets = generate_data(800)
        if test:
            self.data, self.targets = generate_data(200)

        assert len(self.data) > 0

    def __getitem__(self, index):
        input, target = self.data[index], self.targets[index]
        input = torch.from_numpy(input)
        target = torch.from_numpy(target)

        return input, target

    def __len__(self):
        return len(self.data)
