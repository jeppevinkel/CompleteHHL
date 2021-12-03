import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister
from qiskit.extensions import UnitaryGate
from divideAndConquer import DivideAndConquer
from unitaryDecomposition import CUGate, CUGateInverse, UMatrix, MatToEvenHermitian, UGateInverse


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


def create_qft(size: int, printCircuit=False):
    qr = QuantumRegister(size)
    qc = QuantumCircuit(qr, name="qft")
    qft(qc, qr)

    if printCircuit:
        plot = qc.draw(output='mpl')
        plot.show()
        print(qc.draw())
    return qc


def create_qft_inverse(size: int, printCircuit=False):
    qc = create_qft(size).inverse()
    qc.name = "inv_qft"

    if printCircuit:
        plot = qc.draw(output='mpl')
        plot.show()
        print(qc.draw())
    return qc


def ry_rotation(eigTilde, C):
    theta = 2 * np.arcsin(C / eigTilde)
    return theta


def hhl(A, b: np.ndarray, t=np.pi, printCircuit: bool = False):
    # Ensure A is hermitian and b is normalized.
    A, b = MatToEvenHermitian(A, b)
    circuit = QuantumCircuit()
    ancillaRegister = QuantumRegister(1, name='ancilla')
    cRegister = QuantumRegister(A.shape[0], name='clock')
    divideAndConquer = DivideAndConquer(circuit)
    usingDivide: bool = False
    if b.size == 2:
        bRegister = QuantumRegister(1, name='b')
        circuit.add_register(bRegister)
        theta = np.arccos(b[0])
        circuit.ry(theta * 2, bRegister)
    else:
        divideAndConquer.loadB(b)
        bRegister = divideAndConquer.measurePoints
        usingDivide = True
    measurement = ClassicalRegister(bRegister.size + 2, name='measurement')

    circuit.add_register(ancillaRegister)
    circuit.add_register(cRegister)
    circuit.add_register(measurement)

    Umatrix, eigs = UMatrix(A, t=t, debug=printCircuit)
    CU = CUGate(Umatrix)  # DeprecationWarning!
    CU_Inverse = CUGateInverse(Umatrix)  # DeprecationWarning!
    circuit.barrier()
    # for i in range(bRegister.size):
    #     circuit.measure(bRegister[i], measurement[i + 1])
    # ---------QPE------------
    circuit.h(cRegister)

    for k in range(cRegister.size):
        for i in range(np.power(2, k)):
            circuit.append(CU, [cRegister[k], *bRegister])

    # IQFT
    inv_qft = create_qft_inverse(cRegister.size, printCircuit)
    circuit.append(inv_qft, cRegister)
    print("Eigenvalues: " + str(eigs))
    # ---------RY-------------
    eigTilde = (eigs * t / (2 * np.pi)) * 2 ** cRegister.size
    print("Encoded eigenvalues: ", eigTilde)
    C = np.min(np.abs(eigTilde))  # Serching somehow for min... NOT GOOD

    for i in range(cRegister.size):
        # circuit.cry(rY_roation(eigTilde[i], C), cRegister[i], ancillaRegister)
        circuit.cry(ry_rotation(eigTilde[cRegister.size - 1 - i], C), cRegister[i], ancillaRegister)
    circuit.measure(ancillaRegister, measurement[0])

    # --------IQPE-------------
    # QFT
    _qft = create_qft(cRegister.size, printCircuit)
    circuit.append(_qft, cRegister)
    for k in range(cRegister.size):
        for i in range(np.power(2, cRegister.size - 1 - k)):
            circuit.append(CU_Inverse, [cRegister[cRegister.size - 1 - k], *bRegister])

    circuit.barrier()
    circuit.h(cRegister)

    # Measurements-----------------
    for i in range(bRegister.size):
        circuit.measure(bRegister[i], measurement[i + 1])

    # HHL finished!-------------------------
    if printCircuit == True:
        circuit.draw(output='mpl').show()
        print(circuit.draw())

    return circuit
