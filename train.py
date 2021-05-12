"""
Created on April 15 2021

@author: Caleb Sanders

Optimize a QNADE network to the ground state, solve for the ground state energy.    
"""

import matplotlib.pyplot as plt
from QNADE import *
from utils import generate_possible_states, calculate_epsilons, TFIM_exact

# define device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# number of qubits
L = 5
H = 2*L

J = 1 #Ising activation
B = 1 #magnetic field coupling strength

# initialize network and model, put network on device 
network = nn.Sequential(nn.Linear(L,H), nn.Tanh(), nn.Linear(H,2), nn.Tanh())
network.to(device) 
model = QNADE(network)
params = list(model.parameters())

# Training hyperparameters 
iters = 100
batch_size = 1000
lr = 0.005

# initialize optimizer 
optimizer = torch.optim.Adam(params=network.parameters(), lr=lr)
optimizer.zero_grad()

# training loop 
energies = []
for iter in range(iters):

  if iter%10 == 0: print("iter: {}".format(iter))

  # generate data 
  psi_omega, samples, grads_per_param = model(N_samples=batch_size)

  # calculate local energies 
  epsilons = calculate_epsilons(model, samples, psi_omega, B, J).to(device)

  # E is an average of local energies
  E = epsilons.mean()
  energies.append(E)
  epsilons -= E 
  
  # calculate O_k for a given parameter and number of samples 
  for param in range(len(params)):

    # define O_k for a set of parameters 
    O_k = grads_per_param[param].detach()

    # weight O_k according to epsilons 
    O_k *= epsilons.reshape(epsilons.shape + (1,)*(O_k.ndim-1))

    # e_grad is an average of all O_k_s 
    e_grad = torch.mean(O_k, 0, keepdim=True).squeeze()

    # update network parameter matrix with energy gradient  
    with torch.no_grad():
      params[param].grad.copy_(e_grad)

  #optimize network based on e_grad 
  optimizer.step() 

final_energy = min(energies)
print("QNADE Energy: {}".format(final_energy))

# plot training data 
plt.figure()
plt.title("L={}; g={}; QNADE Energy={}".format(L,B,final_energy))
plt.plot(energies)

if L<12:
  expected_e = TFIM_exact(L,B)
  expected_e_plot = [expected_e for i in range(iters)]
  print("Expected Energy: {}".format(expected_e))
  plt.plot(expected_e_plot, 'g--')

plt.xlabel("Iterations")
plt.ylabel("Energy")
plt.savefig("Figure.png")
plt.show()