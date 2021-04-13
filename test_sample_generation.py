"""
Created April 13 2021

@author: Caleb Sanders

This file tests the generation of samples according to the NADE algorithm.
Samples are generated and plotted alongside their theoretical distribution. 
"""

def generate_possible_states(n):
    """
    Generate a numpy array of possible states for a system of n qubits 
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

    return np.array(possible_states)


probs = WAVS**2

possible_states = generate_possible_states(L_in)
bins = np.arange(start=1, stop=(2**L_in)+1, step=1)

sampled_hist_data = [] #sampled state counts for each possible states 
sample_prob_indices = [0]*len(possible_states)

# re-bin and organize the sampled data 
for pos_state in range(len(possible_states)):

    count = 0
    for state in range(len(samples)):
        if np.array_equal(samples[state], possible_states[pos_state]):
            count += 1
            sample_prob_indices[pos_state] = probs[state]

    sampled_hist_data.append(float(count))

sampled_hist_data = np.asanyarray(sampled_hist_data)
sampled_hist_data /= sum(sampled_hist_data)


plt.bar(bins, sampled_hist_data, alpha=0.3, label='samples')
plt.scatter(bins, sample_prob_indices, label='|Psi|^2')
plt.title("System Size: {}".format(L_in))
plt.xlabel("Possible States")
plt.ylabel("Normalized Sample Count")
plt.legend()
plt.savefig("sample_gen_ex.png")
plt.show()