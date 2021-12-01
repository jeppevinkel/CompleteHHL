import numpy as np
from scipy.linalg import lu_factor, lu_solve

def LUsolve(data, n):
    """
    :param data: System of linear equations.
    :param n: Number of variables in equation.
    """
    y_data, x_data = data.T
    # print('\nLU - Decomposition')
    A = x_data.reshape((-1, 1)) ** np.arange(n)

    C = A.T @ A
    c = A.T @ y_data

    (lu, piv) = lu_factor(C)
    x = lu_solve((lu, piv), c)
    # print('lu solution:', x)
    return x
