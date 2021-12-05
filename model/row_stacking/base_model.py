import torch
import torchvision.datasets as datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, OneHotEncoder

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM as mixedglm


import os
os.chdir('../')

import sys  
sys.path.insert(0, '../../')
from utils import *
from inference import *

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# load the training data
mnist_train = datasets.FashionMNIST('../data', train=True, download=True,
                                    transform = transform)

mnist_test = datasets.FashionMNIST('../data', train=False, download=True,
                                    transform = transform)

mnist_train = list(mnist_train)
mnist_test = list(mnist_test)

item = {
    0: ['T-shirt', 'Top'],
    1: ['Trouser', 'Bottom'],
    2: ['Pullover', 'Top'],
    3: ['Dress', 'Bottom'],
    4: ['Coat', 'Top'],
    5: ['Sandal', 'Shoe'],
    6: ['Shirt', 'Top'],
    7: ['Sneaker', 'Shoe'],
    8: ['Bag', 'Bag'],
    9: ['Ankle boot', 'Shoe']
}

    
# row-stacking
ds1_size = 0.5
ds2_size = 0.5
ds3_size = 0.5
batch_size = 64
std_list = [0.1, 0.2, 0.1]

test_size = [1-ds1_size, 1-ds2_size, 1-ds3_size]

# biased toward target
category_wt1 = {
    'Top': [0.5],
    'Bottom': [0.5],
    'Shoe': [0.9],
    'Bag': [0.9]
}

# unbiased
category_wt2 = {
    'Top': [0.9],
    'Bottom': [0.9],
    'Shoe': [0.9],
    'Bag': [0.9]
}

# biased toward non-target
category_wt3 = {
    'Top': [0.9],
    'Bottom': [0.9],
    'Shoe': [0.3],
    'Bag': [0.3]
}

category_wt = [category_wt1, category_wt2, category_wt3]


(obs_X_list, obs_y_list, nonobs_X_list, nonobs_y_list, rest_X_list, rest_y_list) = generate_data_mixture_base(0, mnist_train, 
                                                       test_size, 
                                                       std_list, 
                                                       category_wt, 
                                                       item, batch_size)


trainX_selected = obs_X_list[0]
trainy_selected = obs_y_list[0]


obs_data = define_category(trainy_selected, category_wt1, item).drop('selection_wt', axis = 1)
obs_data['true_pct'] = 0.1
obs_data = obs_data.set_index('item')


# row-stacking 
plot1(0.5, 0.5, std_list)

# weighted-row-stacking
plot2(0.5, 0.5, std_list)