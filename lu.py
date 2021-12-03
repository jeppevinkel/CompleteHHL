import scipy.linalg


def lu_solve(A, b):
    """
    :param data: System of linear equations.
    :param n: Number of variables in equation.
    """

    C = A.T @ A
    c = A.T @ b

    (lu, piv) = scipy.linalg.lu_factor(C)
    x = scipy.linalg.lu_solve((lu, piv), c)
    return x
