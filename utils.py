"""
Created on April 13 2021

@author: Caleb Sanders 

Utility functions. 
"""

import torch
import math 
import numpy as np 

def generate_possible_states(n):
    """
    Generate tensor of possible states for n qubits 
    """

    # 2^(n-1)  2^n - 1 inclusive
    bin_arr = range(0, int(math.pow(2,n)))
    bin_arr = [bin(i)[2:] for i in bin_arr]

    # Prepending 0's to binary strings
    max_len = len(max(bin_arr, key=len))
    bin_arr = [i.zfill(max_len) for i in bin_arr]

    possible_states = []
    for bit_string in bin_arr:
        state = []
        for char in bit_string:
            bit = float(char)
            if bit == 0.0:
                state.append(-1.0)
            else:
                state.append(bit)
        possible_states.append(state)

    return torch.tensor(possible_states, dtype=torch.float)

def calculate_epsilons(model, s, psi_omega, B, J):
        """
        Calculates the E_loc(s) for all sampled states according to the TFIM: 
            epsilon(s) = sum(s_i * s_i+1) + B/psi_s * sum(psi_s_prime).
        Method could be expanded to include more Hamiltonians.

        Args: 
            model: QNADE model 
            s: sampled states matrix
            psi_omega: np list of wavefunction coefficients    
            B: int, sigma_x activation
            J: int, sigma_z activation 

        Returns: 
            epsilon: double, epsilon contribution for the given state 
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        N = len(s[0])
        
        z_term = torch.zeros([len(s)]).to(device)

        # sum of the wavefunction coefficients resulting from sigma_x acting on each qubit (per sample)
        psi_s_prime_sum = torch.zeros([len(s)]).to(device)

        for i in range(N):
            
            if i == N-1:
                z_term += s[:,0]*s[:,i]
            else:
                z_term += s[:,i]*s[:,i+1]

            # calculate the sum of psi_s_prime for the sigma_x term
            s_prime = s.clone()
            s_prime[:,i] = -1*s_prime[:,i]
            psi_s_prime,_,_ = model(x=s_prime, requires_grad=False) 
            psi_s_prime_sum += psi_s_prime
        
        x_term = psi_s_prime_sum/psi_omega

        epsilons = -(J*z_term + B*x_term)
        
        return epsilons

def TFIM_exact(n, g):

    """
    Exact ground state energy calculation according to TFIM. 

    Args: 
        n: number of qubits (int)
        g: magnetic field coupling strenth (float)
    
    Returns: 
        energy: ground state energy (float)
    """

    J = 1
    B = g

    #Initialize operators and Identity matrix
    s_z = np.array([[1,0],[0,-1]])
    s_x = np.array([[0,1],[1,0]])
    I = np.array([[1,0],[0,1]])

    #Calculate and print energies for n qubits
    H = np.zeros((2**n,2**n))
    H_ising = np.zeros((2**n,2**n))
    H_tf = np.zeros((2**n,2**n))

    #Generate and add Ising components to Hamiltonian
    for i in range(0,n):
        if i == 0 or i == n-1:
            ising_comp = s_z
        else:
            ising_comp = I
        for j in range(0,n-1):
            if j == i or j == i-1:
                ising_comp = np.kron(ising_comp, s_z)
            else:
                ising_comp = np.kron(ising_comp, I)
        H_ising += ising_comp

    #Generate and add transverse components to Hamiltonian
    for i in range(0,n):
        if i == 0:
            trans_comp = s_x
        else:
            trans_comp = I
        for j in range(0,n-1):
            if j == i-1:
                trans_comp = np.kron(trans_comp, s_x)
            else:
                trans_comp = np.kron(trans_comp, I)
        H_tf += trans_comp
        
    H = -(J*H_ising + B*H_tf)

    return min(np.linalg.eigvals(H))