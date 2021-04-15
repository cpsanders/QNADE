"""
Created on April 15 2021

@author: Caleb Sanders

Optimize a network to the ground state and solve for the ground state 
energy of an arbitrary-sized quantum many-body system.   
"""

import matplotlib.pyplot as plt
from QNADE import * 
from utils import generate_possible_states, calculate_epsilons

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

B = 1 #sigma_x activation
J = 1 #sigma_z activation

L_in = 2 #number of qubits
H = 2*L_in #number of hidden nodes 
iters = 50 #training loop iterations 
batch_size = 1000 #number of samples 
lr = 0.01
energies = []

# flag to specify sampling/no-sampling optimization 
sampling = True
if sampling == False: hc_samples = generate_possible_states(L_in)

# initialize network, model, store initial parameters 
network = nn.Sequential(nn.Linear(L_in,H), nn.Tanh(), nn.Linear(H,2), nn.Tanh())
model = QNADE(network)
params = list(model.parameters())

# initialize optimizer 
optimizer = torch.optim.Adam(params=network.parameters(), lr=lr)
optimizer.zero_grad()

# training loop 
for iter in range(iters):

  print("iter: {}".format(iter))

  # sampling optimization
  if sampling == True:

    # generate data 
    psi_omega, samples, grads_per_param = model(N_samples=batch_size)

    # calculate local energies 
    epsilons = calculate_epsilons(model, samples, psi_omega, B, J).numpy()

    # E is an average of local energies
    E = sum(epsilons)/len(epsilons)
    energies.append(E)
    epsilons -= E
    
    # calculate O_k for a given parameter and number of samples 
    for param in range(len(params)):
      O_k = grads_per_param[param]
      # re-shape epsilons for multiplication purposes 
      reshaped_epsilons = epsilons.reshape(epsilons.shape + (1,)*(O_k.ndim-1))
      O_k *= reshaped_epsilons  
      # e_grad is an average of all O_k_s 
      e_grad = torch.tensor(sum(O_k)/len(O_k))
      # update network parameter matrix with energy gradient  
      with torch.no_grad():
        params[param].grad.copy_(e_grad)

  # no-sampling optimization 
  else:

    # generate data for hardcoded samples 
    psi_omega, _, grads_per_param = model(x=hc_samples, requires_grad=True)
    psi_omega = psi_omega.numpy()

    # calculate local energies 
    epsilons = calculate_epsilons(model, hc_samples, psi_omega, B, J).numpy()

    # E is a weighted average of epsilons based on respective probability (psi^2)
    E = np.matmul(epsilons, psi_omega**2)
    energies.append(E)
  
    # calculate O_k for a given parameter and number of samples 
    for param in range(len(params)):
      O_k = grads_per_param[param]
      # re-shape epsilons and psi_omega for multiplication purposes
      reshaped_epsilons = epsilons.reshape(epsilons.shape + (1,)*( O_k.ndim-1))
      reshaped_psis = psi_omega.reshape(psi_omega.shape + (1,)*(O_k.ndim-1))
      # e_grad = <O_k * epsilons> - <O_k> * <epsilons>
      e_grad = sum( (O_k*reshaped_epsilons)*(reshaped_psis**2) ) - sum(O_k*(reshaped_psis**2))*E
      # update network parameter matrix with energy gradient 
      with torch.no_grad():
        params[param].grad.copy_(torch.tensor(e_grad))

  optimizer.step() #optimize network based on e_grad 

final_energy = energies[len(energies)-1]

# plot training data 
print("Final Energy: " + str(final_energy))
plt.figure()
plt.title("Number of qubits: {}".format(L_in))
plt.plot(energies)
plt.xlabel("Iterations")
plt.ylabel("Energy")
plt.show()