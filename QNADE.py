#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Jan 28 2021

@authors: Alex Lidiak, Caleb Sanders
This model takes as input a FFNN (N inputs, 2 outputs) and converts it into a 
QNADE model. 

The QNADE class performs the autoregressive sample generation, conditional 
wavefunction calculation, and gradient accumulation needed to optimize a 
FFNN to produce the ground state energy of an arbitrary many-body system.  
"""

import torch
import math 
import autograd_hacks
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 

class QNADE(nn.Module): # takes a FFNN model as input
            
    def __init__(self, model): 
        super(QNADE, self).__init__()
        
        self.model = model
        self.D = self.model[0].in_features # input layer size
        self.M = self.model[-2].out_features # output layer size
        self.evals = [0,1]
            
    def forward(self, N_samples=None, x=None, requires_grad=True):

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.to(device)

        if N_samples is None and x is None: 
            raise ValueError('Must enter samples or the number of samples to' \
                             ' be generated')
            
        if N_samples is None and x is not None: 
            N_samples, need_samples = x.shape[0], False

        if N_samples is not None and x is None: 
            need_samples = True 
            x = torch.zeros([N_samples,self.D],dtype=torch.float)#.to(device)
            
        # the full wvfxn is a product of conditional wvfxns 
        WAV = torch.ones([N_samples])#.to(device) 
        order = np.arange(0,self.D) # sequential autoregressive ordering 
        
        params = list(self.parameters())
        grads_per_param = []
        
        for d in range(self.D):
                
            # masks enforce the autoregressive property
            mask=torch.zeros_like(x)
            mask[:,order[0:(d)]]=1 

            # add autograd hooks for per-sample gradient tracking
            if not hasattr(self.model,'autograd_hacks_hooks'):             
              autograd_hacks.add_hooks(self.model)
            
            # L2 normalization of masked output
            out = F.normalize(self.model(mask*x), 2)
            
            # psi_pos -> positive bits; psi_neg -> negative bits 
            psi_pos = out[:,0].squeeze()
            psi_neg = out[:,1].squeeze()
            
            # Sampling probability is determined by separate conditionals
            if need_samples == True:
                
                # sampling routine according to psi**2:
                m = torch.distributions.Bernoulli(psi_pos**2).sample()

                # convert bit values from 0 to -1 BOTTLENECK 
                for bit in m:
                  if bit == 0.: bit.copy_(torch.tensor(-1, dtype=torch.float))
                
                # update sample tensor
                x[:,d] = m

                # Accumulate PSI based on which state (s) was sampled
                selected_wavs = torch.where(x[:,d] > 0, psi_pos, psi_neg) 
                WAV = WAV*selected_wavs

            else:
                
                # if not sampling, m is a list of bits in column d 
                m = x[:,d]

                # Accumulate PPSI based on which state (s) was sampled
                selected_wavs = torch.where(m > 0, psi_pos, psi_neg) 
                WAV = WAV*selected_wavs

            if requires_grad == True:

              # GRADIENT CALCULATION 
              # eval_grads stores backpropagation values for out1 and out2.
              # eval_grads[0] are the out1 grads for all samples (per param), 
              # eval_grads[1] are the out2 grads for all samples (per param). 
              eval_grads = [ [[]]*len(params) for outputs in range(len(self.evals)) ]

              # Store the per-output grads in eval_grads
              for output in range(len(self.evals)):

                # backpropagate the current output (out1 or out2)
                out[:,output].mean(0).backward(retain_graph=True)

                # compute gradients for all samples 
                autograd_hacks.compute_grad1(self.model)
                autograd_hacks.clear_backprops(self.model)
                
                # store the calculated gradients for all samples 
                for param in range(len(params)):
                  eval_grads[output][param] = params[param].grad1.numpy()

              # allocate space for gradient accumulation 
              if d == 0:
                for param in range(len(params)):
                  grads_per_param.append(np.zeros_like(eval_grads[0][param]))
              
              # accumulate gradients per parameter based on sampled bits; BOTTLENECK
              for sample in range(len(x)):
                for param in range(len(params)):
                  if m[sample] > 0:
                    grads_per_param[param][sample] += eval_grads[0][param][sample]/selected_wavs[sample].detach().numpy()
                  else:
                    grads_per_param[param][sample] += eval_grads[1][param][sample]/selected_wavs[sample].detach().numpy()
                
        return WAV.detach(), x.detach(), grads_per_param