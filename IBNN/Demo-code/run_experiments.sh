#!/bin/bash

mode='train'
lr=0.001


# Train different BNNs on the pancreas data
# python3.9 bayesian_net.py --lr=$lr --mode='train'  --seed=1 --save-dir='./models/AP/seed_1'
# python3.9 bayesian_net.py --lr=$lr --mode='train'  --seed=2 --save-dir='./models/AP/seed_2'
# python3.9 bayesian_net.py --lr=$lr --mode='train'  --seed=3 --save-dir='./models/AP/seed_3'
# python3.9 bayesian_net.py --lr=$lr --mode='train'  --seed=4 --save-dir='./models/AP/seed_4'
#


# There are pretrained models in the current './models/' folder. 


# Average regression error of individual BNNs on test data.
python3.9 bayesian_net.py --mode='test' --seed=1 --save-dir='./models/AP/seed_1'
python3.9 bayesian_net.py --mode='test' --seed=2 --save-dir='./models/AP/seed_2'
python3.9 bayesian_net.py --mode='test' --seed=3 --save-dir='./models/AP/seed_3'
python3.9 bayesian_net.py --mode='test' --seed=4 --save-dir='./models/AP/seed_4'


# Checks inclusion ground truth inside the prediction set.
python3.9 execute-ibnn.py
