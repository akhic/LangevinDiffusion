import numpy as np
import matplotlib.pyplot as plt

def generate_fourier_basis_function(coefficients, period=2 * np.pi):
    """
    Generate a function represented on a Fourier basis.

    Parameters:
    coefficients (np.array): Coefficients of the Fourier terms.
    gradient (bool) : whether to return gradient
    period (float): Period of the function.

    Returns:
    callable: A function represented on a Fourier basis.
    Useless GPT code that I have to rewrite
    """

    num_terms = len(coefficients)//2
    def fourier_basis_function(x):
        k = np.arange(1, num_terms+1).reshape(num_terms, 1)/period
        np.dot(coefficients[:num_terms].T, np.sin(np.dot(k, x.T))).T + np.dot(coefficients[num_terms:].T, np.cos(np.dot(k, x.T))).T


        # result = np.zeros_like(x)
        # for i in range(num_terms):
        #     result += coefficients[i] * np.sin((i + 1) * x / period)
        # return result

    return fourier_basis_function


if __name__ == '__main__':
    num_terms = 100
    coefficients = np.array([1/i**2 for i in range(1,num_terms+1)]).reshape(num_terms,1)
    coefficients = np.array([1/i**2 for i in range(1,num_terms+1)]).reshape(num_terms,1)
    f = generate_fourier_basis_function(coefficients)
    # Generate synthetic data
    # g^-1 = A'

    n = 100_000
    alpha = 2
    period = 2 * np.pi
    xx = np.arange(0, 50,0.1)
    xx = xx.reshape(xx.shape[0],1)
    y = f(xx)
    # np.random.seed(42)
    # X = 10000 * (np.random.rand(n, 1)- 0.5)
    # y = (logistic(f(X)) +  np.random.randn(n, 1)/50 > 0.5).astype(int)