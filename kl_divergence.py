import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def update_vector(old_theta,occurrence,learning_rate):
  gradient =np.array( occurrence)/np.array(old_theta)
  current_theta = old_theta + learning_rate * gradient
  current_theta/=current_theta.sum()
  return current_theta

def kl_divergence(p, q):
    """
    increase in one direction relevant to the observation that we have 
    """
    p = np.array(p)
    q = np.array(q)
    # Compute KL divergence
    return np.sum(p * np.log(p / q))






def generate_samples(n, p_star):
    # Define the possible outcomes as rows of X
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=int)
    
    # Define the probabilities of each outcome (example: p_star = [0.1, 0.3, 0.2, 0.4])
    outcomes = [0, 1, 2, 3]
    
    # Generate random sample indices from p_star
    samples = np.random.choice(outcomes, size=n, p=p_star)
    samples = np.array(samples)
    print(samples)
    # Map indices to corresponding rows in X
    mapped_samples = X[samples]
    # Print for demonstration delete we need the true distribution for the plotting 
    list = []
    for i in range(4):
        count = 0
        for j in samples:
            if j == i:
                count += 1
        list.append(count / n)

    print(list)
    list = np.array(list)
    
    return samples, mapped_samples

# rename it in a better convention 
p_star = [0.4, 0.3, 0.1, 0.2]
p_theta = [0.25, 0.25, 0.25, 0.25]  # this is the true probability 
n = 10000  # Number of samples to generate
# generate the samples and the probability distribution
samples, mapped_samples = generate_samples(n, p_star)
learning_rate =  1e-5 
kl_values = [] 
i = 0
arr = [0, 1, 2, 3]
max_iters = 500
occurrence = [np.sum(samples == el) for el in arr]
print(occurrence)
p_theta_history = [[] for _ in range(4)]  

while kl_divergence(p_theta, p_star) >= 0.00001 and i < max_iters :

    p_theta = update_vector(p_theta,occurrence,learning_rate)
    kl_values.append(kl_divergence(p_theta, p_star))
    for j in range(4):
        p_theta_history[j].append(p_theta[j])  # Store each component
    print(p_theta)
    i += 1





#print(f'In the {i}th iteration the kl divergence converges to, {kl_divergence(p_theta, p_star)},and p_theta={p_theta}')    


# Plot the KL divergence
plt.figure(figsize=(10, 5))
plt.plot(range(i), kl_values, label="KL Divergence ")
plt.xlabel("Iterations")
plt.ylabel("KL Divergence")
plt.title("KL Divergence Over Iterations")
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 5))
colors = ["blue", "red", "green", "purple"]
for j in range(4):
    plt.plot(range(i), p_theta_history[j], label=f"p_theta[{j}]", color=colors[j])

plt.xlabel("Iterations")
plt.ylabel("Probability")
plt.title("Convergence of Individual p_theta Components")
plt.legend()
plt.grid(True)
plt.show()