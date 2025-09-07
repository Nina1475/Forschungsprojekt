import numpy as np

# Data (your binary string)
#X = np.array([0,0,0,0,0,1,1,1,0,1,0,1,0,1,0,1])
X = np.array([1,0,1,0,1,0,0,1,1,0,1,0,1,1,0,0])


# Hyperparameters
eta = 0.05          # learning rate
epochs = 2000       # training iterations
batch_size = 4      # size of mini-batch
eps = 1e-6          # to avoid log(0)

# Initialize parameters randomly in (0,1)
theta0 = np.random.rand()
theta1 = np.random.rand()
phi0   = np.random.rand()
phi1   = np.random.rand()

# --- Helper functions ---
def compute_gradients(batch, theta0, theta1, phi0, phi1):
    n0 = np.sum(batch == 0)
    n1 = np.sum(batch == 1)
    m = len(batch)

    # --- reconstruction term ---
    d_theta0 = (n0/m)*(-(1-phi0)/(1-theta0+eps)) + (n1/m)*((1-phi1)/(theta0+eps))
    d_theta1 = (n0/m)*(-(phi0)/(1-theta1+eps))   + (n1/m)*((phi1)/(theta1+eps))
    d_phi0   = (n0/m)*(np.log(1-theta1+eps) - np.log(1-theta0+eps))
    d_phi1   = (n1/m)*(np.log(theta1+eps) - np.log(theta0+eps))

    # --- KL term ---
    if n0 > 0:
        d_phi0 -= (n0/m)*(np.log(phi0+eps) - np.log(1-phi0+eps))
    if n1 > 0:
        d_phi1 -= (n1/m)*(np.log(phi1+eps) - np.log(1-phi1+eps))

    return d_theta0, d_theta1, d_phi0, d_phi1

def clamp(x):
    return np.clip(x, eps, 1-eps)

def model_distribution(theta0, theta1):
    """Compute p_theta(x=0), p_theta(x=1) with prior p(z)=0.5"""
    p_x0 = 0.5*((1-theta0) + (1-theta1))
    p_x1 = 0.5*(theta0 + theta1)
    return np.array([p_x0, p_x1])

def marginal_distribution(X):
    """Empirical distribution p*(x) from dataset"""
    n0 = np.sum(X==0)
    n1 = np.sum(X==1)
    n = len(X)
    return np.array([n0/n, n1/n])

def kl_divergence(p_true, p_model):
    """KL(p_true || p_model)"""
    return np.sum(p_true * (np.log(p_true+eps) - np.log(p_model+eps)))

# Precompute true distribution once
p_true = marginal_distribution(X)

# --- Training loop ---
for epoch in range(epochs):
    batch = np.random.choice(X, size=batch_size, replace=True)

    d_theta0, d_theta1, d_phi0, d_phi1 = compute_gradients(batch, theta0, theta1, phi0, phi1)

    # Update parameters (gradient ascent on ELBO)
    theta0 += eta * d_theta0
    theta1 += eta * d_theta1
    phi0   += eta * d_phi0
    phi1   += eta * d_phi1

    # Clamp to keep in (0,1)
    theta0, theta1, phi0, phi1 = map(clamp, [theta0, theta1, phi0, phi1])

    if epoch % 200 == 0:
        p_model = model_distribution(theta0, theta1)
        kl = kl_divergence(p_true, p_model)
        print(f"Epoch {epoch}: θ0={theta0:.4f}, θ1={theta1:.4f}, φ0={phi0:.4f}, φ1={phi1:.4f}, KL={kl:.4f}")

# --- Final results ---
p_model = model_distribution(theta0, theta1)
kl = kl_divergence(p_true, p_model)
print("\nFinal parameters:")
print(f"θ0={theta0:.4f}, θ1={theta1:.4f}, φ0={phi0:.4f}, φ1={phi1:.4f}")
print("True distribution p*(x):", p_true)
print("Model distribution p_theta(x):", p_model)
print("KL(p* || p_theta):", kl)
