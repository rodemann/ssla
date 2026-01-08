import argparse
import os
import shutil
import time
import sys
import json
import torch
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
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import bayesian_torch.models.bayesian.resnet_variational as resnet
import numpy as np
from bayesian_net import *

class insulin_glucose_model():
    def __init__(self, src):

        self.num_monte_carlo = 20
        self.model_collection = []
        self.model_count = 4

        self.src = src
        for model_id in range(self.model_count):

            model = Simple_FeedForward(20, 1)
            model = model.double()

            filename = os.path.join(self.src + str(model_id+1), "bayesian_pancreas_model.pth")
            checkpoint = torch.load(filename, map_location = torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()

            self.model_collection.append(model)


    def predict_glucose_val(self, G_and_I_array):

        G_and_I_array = np.asarray(G_and_I_array)
        G_and_I_array = torch.from_numpy(G_and_I_array)

        results = []

        for model_id in range(self.model_count):
            output_mc = []
            for mc_run in range(self.num_monte_carlo):
                output, _ = self.model_collection[model_id].forward(G_and_I_array)
                output_mc.append(output.cpu().detach().numpy())


            mean_val = np.mean(output_mc, axis = 0)
            variance_val = np.var(output_mc, axis = 0)
            results.append( {'mean':mean_val, 'variance':variance_val, 'samples_count': self.num_monte_carlo})

        return results

    def test_glucose_inclusion(self):


        test_dataset = diabetes_loader(train = False, test = True)

        ibnn_correct_count = 0.0
        ensemble_correct_count = 0.0

        for i in range(len(test_dataset.data)):
            input = test_dataset.data[i][0]
            target = test_dataset.data[i][1]
            prediction = ensemble_model.predict_glucose_val(input)
            ibnn_prediction_region = self.uncertainty_ibnn(prediction)
            ensemble_prediction_region = self.uncertainty_ensemble(prediction)

            if (ibnn_prediction_region[0] < target) and (ibnn_prediction_region[2] > target) :
                ibnn_correct_count += 1.0

            if (ensemble_prediction_region[0] < target) and (ensemble_prediction_region[2] > target) :
                ensemble_correct_count += 1.0


            # print("For input - ", input, " target - ", target, \
            #       " ibnn prediction region - ", ibnn_prediction_region, \
            #         " ensemble prediction region - ", ensemble_prediction_region)

        print("IBNN is correctness fraction - ", ibnn_correct_count / float(len(test_dataset.data)))
        print("Ensemble is correct - ", ensemble_correct_count / float(len(test_dataset.data)))
    
    


    def uncertainty_ibnn(self, results):
        ratio = 1.96 # 95% confidence interval
        # ratio = 1.645 # 90% confidence interval
        # ratio = 2.58 # 99% confidence interval

        mean_values = []
        high_values = []
        low_values = []
        for result in results:
            std_dev = np.sqrt(result['variance'][0])
            high_val = result['mean'][0] + ratio * std_dev / float(np.sqrt(self.num_monte_carlo))
            low_val = result['mean'][0] - ratio * std_dev / float(np.sqrt(self.num_monte_carlo))
            mean_val = result['mean'][0]

            high_values.append(high_val)
            low_values.append(low_val)
            mean_values.append(mean_val)


        high_values = np.asarray(high_values)
        low_values = np.asarray(low_values)
        mean_values = np.asarray(mean_values)

        high_limit = np.max(high_values)
        low_limit = np.min(low_values)
        mean = np.mean(mean_values)

        return [low_limit, mean, high_limit]


    def uncertainty_ensemble(self, results):
        ratio = 1.96 # 95% confidence interval
        # ratio = 1.645 # 90% confidence interval
        # ratio = 2.58 # 99% confidence interval

        mean_values = []
        variance_values = []

        for result in results:
            mean_values.append(result['mean'][0])
            variance_values.append(result['variance'][0])

        prediction = np.mean(mean_values)
        variance = np.mean(variance_values) + np.var(mean_values)
        std_dev = np.sqrt(variance)

        high_val = prediction + ratio * std_dev / float(np.sqrt(self.num_monte_carlo))
        low_val = prediction - ratio * std_dev / float(np.sqrt(self.num_monte_carlo))


        return [low_val, prediction, high_val]


class insulin_glucose_model_varying_width():
    def __init__(self, src):

        self.num_monte_carlo = 20
        self.model_collection = []
        self.model_count = 4

        width_list = [20,15,10,5]

        self.src = src
        for width in width_list:

            model = Simple_FeedForward(20, 1, width)
            model = model.double()

            filename = os.path.join(self.src + str(width), "bayesian_pancreas_model.pth")
            checkpoint = torch.load(filename, map_location = torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])

            self.model_collection.append(model)


    def predict_glucose_val(self, G_and_I_array):

        G_and_I_array = np.asarray(G_and_I_array)
        G_and_I_array = torch.from_numpy(G_and_I_array)

        results = []

        for model_id in range(self.model_count):
            output_mc = []
            for mc_run in range(self.num_monte_carlo):
                output, _ = self.model_collection[model_id].forward(G_and_I_array)
                output_mc.append(output.cpu().detach().numpy())


            mean_val = np.mean(output_mc, axis = 0)
            variance_val = np.var(output_mc, axis = 0)
            results.append( {'mean':mean_val, 'variance':variance_val, 'samples_count': self.num_monte_carlo})

        return results

    def uncertainty_ibnn(self, results):
        ratio = 1.96 # 95% confidence interval
        # ratio = 1.645 # 90% confidence interval
        # ratio = 2.58 # 99% confidence interval

        mean_values = []
        high_values = []
        low_values = []
        for result in results:
            std_dev = np.sqrt(result['variance'][0])
            high_val = result['mean'][0] + ratio * std_dev  / float(np.sqrt(self.num_monte_carlo))
            low_val = result['mean'][0] - ratio * std_dev  / float(np.sqrt(self.num_monte_carlo))
            mean_val = result['mean'][0]

            high_values.append(high_val)
            low_values.append(low_val)
            mean_values.append(mean_val)


        high_values = np.asarray(high_values)
        low_values = np.asarray(low_values)
        mean_values = np.asarray(mean_values)

        high_limit = np.max(high_values)
        low_limit = np.min(low_values)
        mean = np.mean(mean_values)

        return [low_limit, mean, high_limit]


    def uncertainty_ensemble(self, results):
        ratio = 1.96 # 95% confidence interval
        # ratio = 1.645 # 90% confidence interval
        # ratio = 2.58 # 99% confidence interval

        mean_values = []
        variance_values = []

        for result in results:
            mean_values.append(result['mean'][0])
            variance_values.append(result['variance'][0])

        prediction = np.mean(mean_values)
        variance = np.mean(variance_values) + np.var(mean_values)
        std_dev = np.sqrt(variance)

        high_val = prediction + ratio * std_dev  / float(np.sqrt(self.num_monte_carlo))
        low_val = prediction - ratio * std_dev  / float(np.sqrt(self.num_monte_carlo))


        return [low_val, prediction, high_val]

if __name__ == '__main__' :

    torch.manual_seed(1) 
    ensemble_model = insulin_glucose_model("./models/AP/seed_")
    ensemble_model.test_glucose_inclusion()

    # ensemble_model = insulin_glucose_model_varying_width()
    # G_and_I = G + I
    # prediction = ensemble_model.predict_glucose_val(G_and_I)
    # print("Result IBNN style - ", ensemble_model.uncertainty_ibnn(prediction))
    # print("Result Ensemble style - ", ensemble_model.uncertainty_ensemble(prediction))
