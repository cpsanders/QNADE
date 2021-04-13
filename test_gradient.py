"""
Created April 13 2021

@author: Caleb Sanders

This file compares the gradient computed in QNADE with the manually calculated theoretical gradient.
"""

returned_grad = grads_per_param[0][0][0][0]
print(returned_grad)
psi_sampled = WAVS[0]
state = samples[0]
epsilon = 0.001

print("\nReturned Gradient Element: {}".format(returned_grad))

# adjust network parameters slightly 
params = list(FFNN.parameters())
with torch.no_grad():
  params[0][0][0].copy_(params[0][0][0] + epsilon)

# select the state you sampled and generate it's wavefunction coeff
psi_adj_list, samples_adj, grads_adj, param_grads_adj = model(N_samples=None, x=samples)
psi_adj = psi_adj_list[0]

manual_grad = (math.log(psi_adj) - math.log(psi_sampled))/epsilon

print("Manually Computed Gradient Element: {}\n".format(manual_grad))