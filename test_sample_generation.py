"""
Created April 13 2021

@author: Caleb Sanders

This file tests the generation of samples according to the NADE algorithm.
Samples are generated and plotted alongside their theoretical distribution. 
"""

import torch 
import torch.nn as nn 
import numpy as np 
import matplotlib.pyplot as plt
from QNADE import QNADE
from utils import generate_possible_states

L_in=5
H = 2*L_in
batch_size = 1000

# Initialize network and model 
FFNN = nn.Sequential(nn.Linear(L_in,H), nn.Tanh(), nn.Linear(H,2), nn.Tanh())
model = QNADE(FFNN)

# generate samples, psis, grads, and probabilities  
WAVS, samples, grads_per_param = model(N_samples=batch_size)
probs = WAVS**2

# generate all theoretical possible states 
possible_states = generate_possible_states(L_in)
bins = np.arange(start=1, stop=(2**L_in)+1, step=1)

sampled_hist_data = [] #sampled state counts for each possible states 
sample_prob_indices = [0]*len(possible_states)

# re-bin and organize the sampled data for plotting
for pos_state in range(len(possible_states)):
    count = 0
    for state in range(len(samples)):
        if np.array_equal(samples[state], possible_states[pos_state]):
            count += 1
            sample_prob_indices[pos_state] = probs[state].numpy()
            
    sampled_hist_data.append(float(count))

sampled_hist_data = np.asanyarray((sampled_hist_data))
sampled_hist_data /= sum(sampled_hist_data) 

plt.bar(bins, sampled_hist_data, alpha=0.3, label='samples')
plt.scatter(bins, sample_prob_indices, label='|Psi|^2')
plt.title("System Size: {}".format(L_in))
plt.xlabel("Possible States")
plt.ylabel("Normalized Sample Count")
plt.legend()
plt.show()