import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def update_vector(p_theta, i, learning_rate):
    vector = np.zeros(4)  # Creates a vector [0, 0, 0, 0]
    vector[i] = 1  
    p_theta = p_theta + learning_rate * (1 / p_theta[i]) * vector
    p_theta = p_theta / p_theta.sum() 
    return p_theta

def kl_divergence(p, q):
    """
    increase in one direction relevant to the observation that we have 
    """
    p = np.array(p)
    q = np.array(q)
    # Compute KL divergence
    return np.sum(p * np.log(p / q))

# Calculate the true loss as the expectation of the loss function over the true distribution.
def true_loss(p_theta, p_star, loss_function):
    return sum(p * loss_function(p_theta, i) for i, p in enumerate(p_star))

# Function to calculate the empirical risk
# Calculate the empirical risk as the average loss over the samples.
def empirical_risk(p_theta, samples, loss_function):
    return np.mean([loss_function(p_theta, sample) for sample in samples])

# Loss function example (KL divergence)
def loss_function(p_theta, i):
    # Example loss function for KL divergence.
    return -np.log(p_theta[i])  # Negative log likelihood for categorical distribution

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
p_theta = [0.4, 0.3, 0.1, 0.2]
p_star = [0.25, 0.25, 0.25, 0.25]  # this is the true probability 
n = 10000  # Number of samples to generate
# generate the samples and the probability distribution
samples, mapped_samples = generate_samples(n, p_star)
learning_rate = 0.001
kl_values = [] 
i = 0
while kl_divergence(p_theta, p_star) >= 0.00001:
    p_theta = update_vector(p_theta, samples[i], learning_rate)
    kl_values.append(kl_divergence(p_theta, p_star))
    print(p_theta)
    i += 1
print(f'In the {i}th iteration the kl divergence converges to, {kl_divergence(p_theta, p_star)},and p_theta={p_theta}')    

used_samples = samples[:i]
true_loss_value = true_loss(p_theta, p_star, loss_function)
empirical_risk_value = empirical_risk(p_theta, used_samples, loss_function)

print(f"True Loss: {true_loss_value}")
print(f"Empirical Risk: {empirical_risk_value}")

# Plot the KL divergence
plt.figure(figsize=(10, 5))
plt.plot(range(i), kl_values, label="KL Divergence ")
plt.xlabel("Iterations")
plt.ylabel("KL Divergence")
plt.title("KL Divergence Over Iterations")
plt.legend()
plt.grid(True)
plt.show()
