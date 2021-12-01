import numpy as np
from qiskit.extensions import UnitaryGate


def isHermitian(matrix: np.ndarray):
    return matrix.ndim == 2 and np.array_equal(matrix, matrix.T)


def isNQubitOperator(matrix: np.ndarray):
    return matrix.shape[0] == matrix.shape[1] and np.mod(np.log(matrix.shape[0]) / np.log(2), 1) == 0


def getNextN(matrix: np.ndarray):
    n = matrix.shape[0]
    minTarget = (matrix.shape[0] + matrix.shape[1])
    while not np.mod(np.log(n) / np.log(2), 1) == 0 or n < minTarget:
        n = n + 1
    return n - minTarget


def MatToEvenHermitian(a: np.ndarray, b: np.ndarray):
    assert a.ndim == 2, "The A matrix needs to be 2 dimensional!"

    if isHermitian(a) and a.shape[0] % 2 == 0:
        print("PERFECTION")
        return a, b / np.linalg.norm(b)
    elif not isNQubitOperator(a):
        n = getNextN(a)
        for i in range(n):
            a = np.append(a, np.array([a[-1, ...]]), axis=0)
            b = np.append(b, np.array([b[-1]]), axis=0)

    if not isHermitian(a) and a.shape[0] == a.shape[1]:
        # print("A", a)
        # print("b", b)

        b = a.T @ b
        a = a.T @ a
        # print("A", a)
        # print("b", b)
        return a, b / np.linalg.norm(b)

    # print("A.shape:", a.shape)

    at = a.T
    ht = np.hstack((np.zeros((a.shape[0], at.shape[1])), a))
    hb = np.hstack((at, np.zeros((at.shape[0], a.shape[1]))))

    h = np.vstack((ht, hb))
    b = np.vstack((b, np.zeros((h.shape[0] - b.shape[0], 1))))

    # print(h)
    # print(b)

    return h, b / np.linalg.norm(b)

    if a.shape[0] % 2 == 0 and a.shape[1] % 2 == 0 and not isHermitian(a):
        exit(1337)
    # elif (a.shape[0] % 2 == 1 or a.shape[1] % 2 == 1) and not isHermitian(a):
    #     h = a.T @ a
    #     b = a.T @ b
    #     at = h.T
    #     ht = np.hstack((np.zeros(h.shape), h))
    #     hb = np.hstack((at, np.zeros(at.shape)))
    #     h = np.vstack((ht, hb))
    #     b = np.vstack((b, np.zeros(b.shape)))
    #     return h, b / np.sqrt(np.sum(b ** 2))
    else:
        if not isHermitian(a):
            anew = a.T @ a
            b = a.T @ b
        else:
            anew = a

        print("STUFF")
        return h, b / np.sqrt(np.sum(b ** 2))


def UMatrix(a: np.ndarray, t=np.pi, debug: bool = False):
    # w i eigenvalues, and v is normalized eigenvectors. Each column is an eigenvector.
    print(np.array_equal(a, a.T))
    print("AAAAA", a)
    w, v = np.linalg.eigh(a)
    uDiag = np.zeros((w.size, w.size), dtype=np.complex)
    for i in range(w.size):
        uDiag[i, i] = np.exp(t * w[i] * 1j)
    U: np.ndarray = v @ uDiag @ v.T
    U = np.around(U, 12)

    print(U.shape)

    if debug:
        print('V', v)
        print('Udiag', uDiag)
        print('U', U)
    return U, w


# Gate caching
cachedU: UnitaryGate
cachedUInput: np.ndarray = np.array([])
cachedUInverse: UnitaryGate
cachedUInverseInput: np.ndarray = np.array([])

cachedCU: UnitaryGate
cachedCUInput: np.ndarray = np.array([])
cachedCUInverse: UnitaryGate
cachedCUInverseInput: np.ndarray = np.array([])


def UGate(matrix: np.ndarray):
    global cachedU
    global cachedUInput
    if cachedUInput == matrix:
        return cachedU

    cachedU = UnitaryGate(matrix)
    cachedUInput = matrix
    return cachedU


def UGateInverse(matrix: np.ndarray):
    global cachedUInverse
    global cachedUInverseInput
    if cachedUInverseInput == matrix:
        return cachedUInverse

    cachedUInverse = UGate(matrix).inverse()
    cachedUInverseInput = matrix
    return cachedUInverse


def CUGate(matrix: np.ndarray, num_control_bits: int = 1):
    global cachedCU
    global cachedCUInput
    if np.array_equal(cachedCUInput, matrix):
        return cachedCU

    cachedCU = UnitaryGate(matrix).control(num_control_bits)
    cachedCUInput = matrix
    return cachedCU


def CUGateInverse(matrix: np.ndarray, num_control_bits: int = 1):
    global cachedCUInverse
    global cachedCUInverseInput
    if np.array_equal(cachedCUInverseInput, matrix):
        return cachedCUInverse

    cachedCUInverse = CUGate(matrix, num_control_bits).inverse()
    cachedCUInverseInput = matrix
    return cachedCUInverse
