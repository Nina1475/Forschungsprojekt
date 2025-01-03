import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt



def update_vector(theta,i,learning_rate):
    vector = np.zeros(4)  # Creates a vector [0, 0, 0, 0]
    vector[i] = 1 
    theta = theta + learning_rate * (1 / theta[i]) * vector
    theta = theta / theta.sum() 
    return theta






def kl_divergence(p, q):
    """
    increase in one direction relevant to the observation that we have 
    """
    p = np.array(p)
    q = np.array(q)
    
    # Prevent log(0) and division by 0
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)  # Avoid log(0)
    q = np.clip(q, epsilon, 1)  # Avoid division by 0
    
    # Compute KL divergence
    return np.sum(p * np.log(p / q))




def generate_samples(n, p_star):
    # Define the possible outcomes as rows of X
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=int)
    
    # Define the probabilities of each outcome (example: p_star = [0.1, 0.3, 0.2, 0.4])
    outcomes = [0, 1, 2, 3]
    
    # Generate random sample indices from p_star
    samples = np.random.choice(outcomes, size=n, p=p_star)
    samples= np.array(samples)
    print(samples.sort())
    # Map indices to corresponding rows in X
    mapped_samples = X[samples]
    print(samples)
    # Print for demonstration delete we need the true distribution for the plotting 
    list=[]
    for i in range(4):
        count = 0
        for j in (samples):
            if j==i:
                count+=1
        list.append(count/n)

    print(list)
    list = np.array(list)
    
    return samples, mapped_samples






# rename it in a better convention 
theta = [0.25,0.25,0.25,0.25]
p_true = [0.1, 0.3, 0.2, 0.4]  # this is the true probability 
n = 25  # Number of samples to generate
# generate the samples and the probability distribution
samples, mapped_samples = generate_samples(n,p_true)
print(list)
learning_rate = 0.01
kl_values = [] 
for i in samples:
    theta = update_vector(theta,i,learning_rate)
    kl_values.append(kl_divergence(theta,p_true))
    print(theta)


print (kl_divergence(theta,p_true))
 
# Plot the KL divergence
plt.figure(figsize=(10, 5))
plt.plot(range(25), kl_values, label="KL Divergence ")
plt.xlabel("Iterations")
plt.ylabel("KL Divergence")
plt.title("KL Divergence Over Iterations")
plt.legend()
plt.grid(True)
plt.show()

# github repository 