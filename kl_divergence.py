import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def update_vector(old_theta,occurrence,learning_rate):
  gradient =np.array( occurrence)/np.array(old_theta)
  current_theta = old_theta + learning_rate * gradient
  #current_theta/=current_theta.sum()
  return current_theta

def kl_divergence(p, q):
    """
    increase in one direction relevant to the observation that we have 
    """
    p = np.array(p)
    q = np.array(q)
    # Compute KL divergence
    return np.sum(p * np.log(p / q))


'''
def imperical_ distribution () compute the true distribution of the data samples 
stochatic gradient descent if it doesnt work 
work on latent represenntation and how to inncorrate into the example that we have 

'''
def  imperical_distribution(occurrence):
    return(occurrence/occurrence.sum())



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
#p_star = [0.5, 0.2, 0.1, 0.2]
#p_star = [0.3, 0.4, 0.2, 0.1]
p_theta = [0.25, 0.25, 0.25, 0.25]  # this is the true probability 
n = 10000  # Number of samples to generate
# generate the samples and the probability distribution
samples, mapped_samples = generate_samples(n, p_star)
learning_rate = 5
theta = [[] for _ in range(4)] 
i = 0
arr = [0, 1, 2, 3]
max_iters = 1000
occurrence = [np.sum(samples == el) for el in arr]
#print(occurrence)

for k in range(4):
        theta[k].append(p_theta[k] )


        
while i<=max_iters  :
    
    p_theta = update_vector(p_theta,occurrence,learning_rate)
    for j in range(4):
        theta[j].append((p_theta / p_theta.sum())[j])
    #print(p_theta)
    #for j in range(4):
       #p_theta_history[j].append(p_theta[j])  # Store each component
    print(p_theta)
    i += 1


print( "This is where p_theta connverges",p_theta/ p_theta.sum() )

print("this is the imperical distribution", imperical_distribution(np.array(occurrence)))
print(occurrence)
# Convert to NumPy array for easier slicing

plt.figure(figsize=(10, 6))
colors = ["blue", "red", "green", "purple"]

for j in range(4):
    plt.plot(theta[j][:max_iters], label=f"theta_{j}", color=colors[j])

plt.xlabel('Iteration')
plt.ylabel('Theta values')
plt.title('Convergence of theta over 10000 iterations')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()