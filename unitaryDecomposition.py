import numpy as np
from qiskit.extensions import UnitaryGate


def is_hermitian(matrix: np.ndarray):
    return matrix.ndim == 2 and np.array_equal(matrix, matrix.T)


def is_n_qubit_operator(matrix: np.ndarray):
    return matrix.shape[0] == matrix.shape[1] and np.mod(np.log(matrix.shape[0]) / np.log(2), 1) == 0


def get_next_n(matrix: np.ndarray):
    n = matrix.shape[0]
    min_target = (matrix.shape[0] + matrix.shape[1])
    while not np.mod(np.log(n) / np.log(2), 1) == 0 or n < min_target:
        n = n + 1
    return n - min_target


def mat_to_even_hermitian(a: np.ndarray, b: np.ndarray):
    assert a.ndim == 2, "The A matrix needs to be 2 dimensional!"

    if is_hermitian(a) and a.shape[0] % 2 == 0:
        return a, b / np.linalg.norm(b)
    elif not is_n_qubit_operator(a):
        n = get_next_n(a)
        for i in range(n):
            a = np.append(a, np.array([a[-1, ...]]), axis=0)
            b = np.append(b, np.array([b[-1]]), axis=0)

    if not is_hermitian(a) and a.shape[0] == a.shape[1]:
        b = a.T @ b
        a = a.T @ a
        return a, b / np.linalg.norm(b)

    at = a.T
    ht = np.hstack((np.zeros((a.shape[0], at.shape[1])), a))
    hb = np.hstack((at, np.zeros((at.shape[0], a.shape[1]))))

    h = np.vstack((ht, hb))
    b = np.vstack((b, np.zeros((h.shape[0] - b.shape[0], 1))))

    return h, b / np.linalg.norm(b)


def u_matrix(a: np.ndarray, t=np.pi, debug: bool = False):
    # w i eigenvalues, and v is normalized eigenvectors. Each column is an eigenvector.
    w, v = np.linalg.eigh(a)
    u_diag = np.zeros((w.size, w.size), dtype=np.complex)
    for i in range(w.size):
        u_diag[i, i] = np.exp(t * w[i] * 1j)
    u: np.ndarray = v @ u_diag @ v.T
    u = np.around(u, 12)

    if debug:
        print('V', v)
        print('Udiag', u_diag)
        print('U', u)
        print('EigenValues', w)
    return u, w


# Gate caching
cachedU: UnitaryGate
cachedUInput: np.ndarray
cachedUInverse: UnitaryGate
cachedUInverseInput: np.ndarray

cachedCU: UnitaryGate
cachedCUInput: np.ndarray = np.array([])
cachedCUInverse: UnitaryGate
cachedCUInverseInput: np.ndarray = np.array([])


def u_gate(matrix: np.ndarray):
    global cachedU
    global cachedUInput
    if cachedUInput == matrix:
        return cachedU

    cachedU = UnitaryGate(matrix)
    cachedUInput = matrix
    return cachedU


def u_gate_inverse(matrix: np.ndarray):
    global cachedUInverse
    global cachedUInverseInput
    if cachedUInverseInput == matrix:
        return cachedUInverse

    cachedUInverse = u_gate(matrix).inverse()
    cachedUInverseInput = matrix
    return cachedUInverse


def cu_gate(matrix: np.ndarray, num_control_bits: int = 1):
    global cachedCU
    global cachedCUInput
    if np.array_equal(cachedCUInput, matrix):
        return cachedCU

    cachedCU = UnitaryGate(matrix).control(num_control_bits)
    cachedCUInput = matrix
    return cachedCU


def cu_gate_inverse(matrix: np.ndarray, num_control_bits: int = 1):
    global cachedCUInverse
    global cachedCUInverseInput
    if np.array_equal(cachedCUInverseInput, matrix):
        return cachedCUInverse

    cachedCUInverse = cu_gate(matrix, num_control_bits).inverse()
    cachedCUInverseInput = matrix
    return cachedCUInverse
