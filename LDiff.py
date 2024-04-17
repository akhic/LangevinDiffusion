import numpy as np
import matplotlib.pyplot as plt
from Fourierbasis import generate_fourier_basis_function

def logistic(z):
    return 1 / (1 + np.exp(-z))
 
num_terms = 1000
coefficients = np.array([1/(i//2+1)**3 for i in range(num_terms)]).reshape(num_terms,1)
f = generate_fourier_basis_function(coefficients)
# Generate synthetic data
# g^-1 = A'

n = 1000
alpha = 2
period = 2 * np.pi
# np.random.seed(42)
X = 100 * (np.random.rand(n, 1)- 0.5)
y = (logistic(f(X)) +  np.random.randn(n, 1)/50 > 0.5).astype(int)

# y = (4 + 3 * X + np.random.randn(100, 1) > 5).astype(int)


def prior_log_density(theta, gradient = False):
    p = len(theta)
    k = np.arange(1,p+1).reshape(p,1)
    if gradient:
        # px1 array
        return - n**(1/(2*alpha + 1))* (k**(2*alpha) * theta)
    return -0.5 * n**(1/(2*alpha + 1)) * np.dot(k**(2*alpha).T,(theta**2))

def log_likelihood(theta, X, y, gradient = False):
    ff = generate_fourier_basis_function(theta)
    if gradient:
        p = len(theta)//2
        k = np.arange(1,p+1).reshape(1,p)/period
        diff = (y - logistic(ff(X))).T
        sine_block = np.sin(np.dot(X,k))
        cos_block = np.cos(np.dot(X,k))
        concat_block = np.zeros((n,2*p))
        concat_block[:,::2] = sine_block
        concat_block[:,1::2] = cos_block
        # px1 array
        return np.dot(diff[::2], concat_block).T
        
    log_like = np.dot(y.T, ff(X)) - np.log(1 + np.exp(ff(X))).sum()
    return log_like

# Gradient of log-likelihood function
def grad_log_likelihood(theta, X, y):
    grad_prior = prior_log_density(theta, True)
    grad_like = log_likelihood(theta, X, y, True)
    
    return grad_like + grad_prior

# Unadjusted Langevin algorithm
def langevin_algorithm(num_samples, step_size, initial_theta, X, y):
    theta_current = initial_theta
    samples = [theta_current]
    p = len(initial_theta)
    for i in range(num_samples):
        # Compute the gradient of the log-likelihood
        grad_ll = grad_log_likelihood(theta_current, X, y)

        # Update theta using Langevin dynamics
        theta_proposed = theta_current + step_size * grad_ll + np.sqrt(2 * step_size) * np.random.randn(p, 1)

        # Accept the proposal
        theta_current = theta_proposed
        samples.append(theta_current)

    return np.array(samples)

def gradient_descent(num_iterations, step_size, initial_theta, X, y):
    theta_current = initial_theta
    samples = [theta_current]
    p = len(initial_theta)
    for i in range(num_samples):
        grad_ll = grad_log_likelihood(theta_current, X, y)

        theta_proposed = theta_current + step_size * grad_ll
        theta_current = theta_proposed
        samples.append(theta_current)
    
    return np.array(samples)

# Set hyperparameters
num_samples = 10000
step_size = 0.00001
# initial_theta = np.zeros(10).reshape(10,1)  # Initial guess for coefficients
initial_theta = np.array([1/(i//2+1)**3 for i in range(10)]).reshape(10,1)
# Run Langevin algorithm
samples = langevin_algorithm(num_samples, step_size, initial_theta, X, y)
samples_gd = gradient_descent(num_samples, step_size, initial_theta, X, y)
# Plot the trace and histogram of samples
# plt.figure(figsize=(10, 5))
# plt.subplot(2, 1, 1)
# plt.plot(samples[:, 0], label='Intercept')
# plt.plot(samples[:, 1], label='Coefficient')
# plt.xlabel('Iteration')
# plt.ylabel('Parameter Value')
# plt.title('Trace Plot')
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.hist(samples[:, 0], bins=30, alpha=0.5, label='Intercept')
# plt.hist(samples[:, 1], bins=30, alpha=0.5, label='Coefficient')
# plt.xlabel('Parameter Value')
# plt.ylabel('Frequency')
# plt.title('Histogram of Samples')
# plt.legend()

# plt.tight_layout()
# plt.show()
plt.figure(figsize=(20, 12))
for i in range(10):
    plt.subplot(10, 1, i+1)
    plt.plot(samples[:, i])
    plt.xlabel('Iteration')
    plt.ylabel(f'$\\theta_{i+1}$')
    plt.title(f'Sampling for $\\theta_{i+1}$')
# plt.savefig('PosteriorLangevin.png')
plt.tight_layout()
plt.show()