"""
Created April 13 2021

@author: Caleb Sanders

This file compares the gradient computed in QNADE with the manually calculated theoretical gradient.
"""
import torch
import math 
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt 
from utils import generate_possible_states
from QNADE import QNADE

# input layer size
L_in=2

# hidden layer size
H = 2*L_in

# number of samples 
batch_size = 2

# Initialize network and model 
FFNN = nn.Sequential(nn.Linear(L_in,H), nn.Tanh(), nn.Linear(H,2), nn.Tanh())
model = QNADE(FFNN)

# generate samples, psis, and grads 
WAVS, samples, grads_per_param = model(N_samples=batch_size)

# select a specific psi/grad to look at 
returned_grad = grads_per_param[0][0][0][0]
psi_sampled = WAVS[0]
state = samples[0]
delta = 0.001 #amount by which to adjust a certain parameter 

print("\nReturned Gradient Element: {}".format(returned_grad))

# adjust network parameter slightly 
params = list(FFNN.parameters())
with torch.no_grad():
  params[0][0][0].copy_(params[0][0][0] + delta)

# generate new wavefunction coeff for a specified state based on adjusted param 
psi_adj_list, samples_adj, param_grads_adj = model(N_samples=None, x=samples)
psi_adj = psi_adj_list[0]

# calculate gradient based on definition 
manual_grad = (math.log(psi_adj) - math.log(psi_sampled))/delta

print("Manually Computed Gradient Element: {}\n".format(manual_grad))