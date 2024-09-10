"""
Created on April 15 2021

@author: Caleb Sanders

Optimize a QNADE network to the ground state, solve for the ground state energy.
"""

import logging

import matplotlib.pyplot as plt
import torch
from torch import nn

from qnade.network import QNADE
from qnade.schemas import QNADEConfig
from qnade.utils import calculate_epsilons


def optimize_qnade_network(config: QNADEConfig) -> list[float]:
    """Optimize a QNADE network for a given system configuration."""
    network = nn.Sequential(
        nn.Linear(config.number_of_qubits, config.hidden_layer_dimension),
        nn.Tanh(),
        nn.Linear(config.hidden_layer_dimension, config.hidden_layer_dimension),
        nn.Tanh(),
        nn.Linear(config.hidden_layer_dimension, 2),
        nn.Tanh(),
    )
    network.to(config.device)
    model = QNADE(network)
    params = list(model.parameters())

    optimizer = torch.optim.Adam(params=network.parameters(), lr=config.learning_rate)
    optimizer.zero_grad()

    energies = []
    for iter in range(config.number_of_training_iterations):

        if iter % 10 == 0:
            logging.info("iteration: {}".format(iter))

        # generate data through QNADE sampling
        psi_omega, samples, grads_per_param = model(
            N_samples=config.number_of_training_samples
        )

        # calculate local energies
        epsilons = calculate_epsilons(
            model,
            samples,
            psi_omega,
            config.magnetic_field_coupling_strength,
            config.ising_activation,
        )
        epsilons = epsilons.to(config.device)

        # energy is an average of local energies
        energy = epsilons.mean()
        energies.append(energy)
        epsilons -= energy

        # calculate O_k for a given parameter and number of samples
        for param in range(len(params)):

            # define O_k for a set of parameters
            O_k = grads_per_param[param].detach()

            # weight O_k according to epsilons
            O_k *= epsilons.reshape(epsilons.shape + (1,) * (O_k.ndim - 1))

            # e_grad is an average of all O_k_s
            e_grad = torch.mean(O_k, 0, keepdim=True).squeeze()

            # update network parameter matrix with energy gradient
            with torch.no_grad():
                params[param].grad.copy_(e_grad)

        # optimize network based on e_grad
        optimizer.step()

    return energies
