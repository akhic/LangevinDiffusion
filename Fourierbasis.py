import numpy as np
import matplotlib.pyplot as plt

def generate_fourier_basis_function(coefficients, period=1/ (2 * np.pi)):
    num_terms = len(coefficients)//2
    def fourier_basis_function(x):
        k = np.arange(1, num_terms+1).reshape(num_terms, 1)/period
        return 2**0.5 * np.dot(coefficients[::2].T, np.sin(np.dot(k, x.T))).T + \
                2**0.5 * np.dot(coefficients[1::2].T, np.cos(np.dot(k, x.T))).T


    return fourier_basis_function


if __name__ == '__main__':
    # Sample Usage, the coefficients need to be vstacked
    num_terms = 1000
    coefficients = np.array([1 if i <51 else 1/(i-50)**3 for i in range(1,num_terms+1)]).reshape(num_terms,1)
    
    f = generate_fourier_basis_function(coefficients)
    # Generate synthetic data
    # g^-1 = A'

    n = 100_000
    alpha = 2
    period = 2 * np.pi
    xx = np.arange(-10, 10,0.1)
    xx = xx.reshape(xx.shape[0],1)
    plt.plot(xx, f(xx))
    plt.xlabel('x')
    plt.ylabel('$\\Phi(\\theta)(x)$')
    plt.title('Function represented in Fourier basis')
    plt.savefig('phi.png')
    plt.show()
    