import scipy.linalg

def LUsolve(data, n):
    """
    :param data: System of linear equations.
    :param n: Number of variables in equation.
    """
    y_data, x_data = data.T
    A = x_data.reshape((-1, 1)) ** np.arange(n)

    C = A.T @ A
    c = A.T @ y_data

    (lu, piv) = scipy.linalg.lu_factor(C)
    x = scipy.linalg.lu_solve((lu, piv), c)
    return x
