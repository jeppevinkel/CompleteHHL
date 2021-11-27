import numpy as np
from qiskit.extensions import UnitaryGate


def UGate(matrix: np.ndarray):
    return UnitaryGate(matrix)


def isHermitian(matrix: np.ndarray):
    return np.array_equal(matrix, matrix.T)


def MatToEvenHermitian(a: np.ndarray, b: np.ndarray):
    if a.shape[0] % 2 == 0 and not isHermitian(a):
        h = a.T @ a
        b = a.T @ b
        return h, b / np.sqrt(np.sum(b**2))
    elif isHermitian(a):
        return a, b / np.sqrt(np.sum(b**2))
    else:
        at = a.T
        ht = np.hstack((np.zeros(a.shape), a))
        hb = np.hstack((at, np.zeros(at.shape)))
        h = np.vstack((ht, hb))
        b = np.vstack((b, np.zeros(b.shape)))
        return h, b / np.sqrt(np.sum(b**2))


def UMatrix(a: np.ndarray, t=np.pi, debug: bool = False):
    # w i eigenvalues, and v is normalized eigenvectors. Each column is an eigenvector.
    w, v = np.linalg.eigh(a)
    uDiag = np.zeros((w.size, w.size), dtype=np.complex)
    for i in range(w.size):
        uDiag[i, i] = np.exp(t * w[i] * 1j)
    U: np.ndarray = v @ uDiag @ v.T
    U = np.around(U, 12)

    if debug:
        print('V', v)
        print('Udiag', uDiag)
        print('U', U)

    return U, w


def CUGate(matrix: np.ndarray, num_control_bits: int = 1):
    return UnitaryGate(matrix).control(num_control_bits)
