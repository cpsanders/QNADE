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

      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

      if N_samples is None and x is None: 
          raise ValueError('Must enter samples or the number of samples to' \
                            ' be generated')
          
      # if not sampling, just calculating wavefunction
      if N_samples is None and x is not None: 
          N_samples, need_samples = x.shape[0], False

      # if sampling and calculating wavefunction
      if N_samples is not None and x is None: 
          need_samples = True
          x = torch.zeros([N_samples,self.D],dtype=torch.float).to(device)
          
      # the full wavefunction is a product of the conditionals
      WAV = torch.ones([N_samples]).to(device) 
      order = np.arange(0,self.D) # sequential autoregressive ordering 
      
      # for gradient tracking 
      params = list(self.parameters()) 
      grads_per_param = [] 
      
      for d in range(self.D):
              
          # mask enforces the autoregressive property
          mask=torch.zeros_like(x)
          mask[:,order[0:(d)]] = 1 

          # add autograd hooks for per-sample gradient calculation 
          if not hasattr(self.model,'autograd_hacks_hooks'):             
            autograd_hacks.add_hooks(self.model)
          
          # L2 normalization of masked output
          out = F.normalize(self.model(mask*x), 2)
          
          # 'psi_pos' is positive bits, 'psi_neg' is negative bits 
          psi_pos = out[:,0].squeeze()
          psi_neg = out[:,1].squeeze()

          if need_samples == True:
              
            # sampling routine according to psi**2:
            # convert bit values from 0 to -1
            m = torch.distributions.Bernoulli(psi_pos**2).sample()
            m = torch.where(m == 0, 
                            torch.tensor(-1).to(device), 
                            torch.tensor(1).to(device)) 
            
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
                eval_grads[output][param] = params[param].grad1

            # allocate space for gradient accumulation 
            if d == 0:
              for param in range(len(params)):
                grads_per_param.append(torch.zeros_like(eval_grads[0][param]))
            
            # accumulate gradients per parameter based on sampled bits
            for param in range(len(params)):

              # reshape m and wavs so they can be accumulated/divided properly 
              reshaped_m = m.reshape(m.shape + (1,)*(grads_per_param[param].ndim-1))
              reshaped_wavs = selected_wavs.reshape(selected_wavs.shape + (1,)*(grads_per_param[param].ndim-1))

              # select the proper gradient to accumulate based on m 
              grads_per_param[param][:] += torch.where(reshaped_m[:] > 0, 
                                                        eval_grads[0][param][:]/reshaped_wavs[:], 
                                                        eval_grads[1][param][:]/reshaped_wavs[:])

      return WAV.detach(), x.detach(), grads_per_param