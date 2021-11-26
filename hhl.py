import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.extensions import UnitaryGate
from divideAndConquer import DivideAndConquer
from unitaryDecomposition import CUGate, UMatrix, MatToEvenHermitian


def qft(qc: QuantumCircuit, qr: QuantumRegister):
    n = qr.size
    for i in range(int(n / 2)):
        qc.swap(i, n - 1 - i)
    for q in range(qr.size):
        qb = q
        qc.h(qb)
        r = 2
        i = qb + 1
        while r <= n - q:
            qc.cp(np.pi / 2 ** (r - 1), qb, i)
            r += 1
            i += 1


def create_qft(size: int):
    qr = QuantumRegister(size)
    qc = QuantumCircuit(qr, name="qft")
    qft(qc, qr)

    plot = qc.draw(output='mpl')
    plot.show()
    print(qc.draw())
    return qc


def create_qft_inverse(size: int):
    qc = create_qft(size).inverse()
    qc.name = "inv_qft"

    plot = qc.draw(output='mpl')
    plot.show()
    print(qc.draw())
    return qc


def rY_roation(eigTilde, C):
    return 2 * np.arcsin(C / eigTilde)


def hhl(A, b, t, printCircuit: bool = False):
    # Ensure A is hermitian and b is normalized.
    A, b = MatToEvenHermitian(A, b)

    circuit = QuantumCircuit()
    ancillaRegister = QuantumRegister(1, name='ancilla')
    cRegister = QuantumRegister(A.shape[0], name='clock')
    measurement = ClassicalRegister(b.size()+1, name='measurement')
    divideAndConquer = DivideAndConquer(circuit)
    bRegister = divideAndConquer.loadB(b)

    circuit.add_register(ancillaRegister)
    circuit.add_register(cRegister)
    circuit.add_register(measurement)

    Umatrix, eigs = UMatrix(A)
    CU = CUGate(Umatrix, bRegister.size())

    # ---------QPE------------
    circuit.h(cRegister)
    for k in range(cRegister.size()):
        for i in range(k):
            circuit.append(CU, [cRegister[i], bRegister])

    # circuit.cu(np.pi, 3*np.pi/2, 5*np.pi/2, 0, cRegister[0], bRegister)
    # circuit.cu(np.pi, 3*np.pi/2, 5*np.pi/2, 0, cRegister[1], bRegister)
    # circuit.cu(np.pi, 3*np.pi/2, 5*np.pi/2, 0, cRegister[1], bRegister)
    # IQFT
    inv_qft = create_qft_inverse(cRegister.size)
    circuit.append(inv_qft, cRegister)

    # ---------RY-------------
    eigTilde = (eigs * t / (2 * np.pi)) * 2 ** cRegister.size()
    minEigs = np.min(eigTilde)  # Serching somehow for min... NOT GOOD
    C = (minEigs * t)

    for i in range(cRegister.size()):
        circuit.cry(rY_roation(eigTilde[i], C), cRegister[i], ancillaRegister)

    # --------IQPE-------------
    # QFT
    _qft = create_qft(cRegister.size)
    circuit.append(_qft, cRegister)

    circuit.h(cRegister)
    for k in range(cRegister.size()):
        for i in range(cRegister.size() - 1 - k):
            circuit.append(CU.inverse(), [cRegister[i], bRegister])

    # Measurements-----------------
    circuit.measure(bRegister, measurement[0])
    divideAndConquer.measureB(measurement)

    #HHL finished!-------------------------
    if printCircuit == True:
        circuit.draw(output='mpl').show()