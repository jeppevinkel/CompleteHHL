import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from divideAndConquer import DivideAndConquer
from unitaryDecomposition import cu_gate, cu_gate_inverse, u_matrix, mat_to_even_hermitian


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


def create_qft(size: int, print_circuit=False):
    qr = QuantumRegister(size)
    qc = QuantumCircuit(qr, name="qft")
    qft(qc, qr)

    if print_circuit:
        plot = qc.draw(output='mpl')
        plot.show()
    return qc


def create_qft_inverse(size: int, print_circuit=False):
    qc = create_qft(size).inverse()
    qc.name = "inv_qft"

    if print_circuit:
        plot = qc.draw(output='mpl')
        plot.show()
    return qc


def ry_rotation(eig_tilde, c):
    # print(str(2) + " * np.arcsin(" + str(C) + " / " + str(eigTilde) + ")")
    theta = 2 * np.arcsin(c / eig_tilde)
    # print(theta)
    return theta


def hhl(A, b: np.ndarray, t=np.pi, print_circuit: bool = False):
    # Ensure A is hermitian and b is normalized.
    A, b = mat_to_even_hermitian(A, b)
    circuit = QuantumCircuit()
    ancilla_register = QuantumRegister(1, name='ancilla')
    c_register = QuantumRegister(A.shape[0], name='clock')
    divide_and_conquer = DivideAndConquer(circuit)
    using_divide: bool = False
    if b.size == 2:
        b_register = QuantumRegister(1, name='b')
        circuit.add_register(b_register)
        theta = 2 * np.arccos(b[0])
        print("B", b)
        print("Theta", theta)
        circuit.ry(theta, b_register)
    else:
        divide_and_conquer.load_b(b)
        b_register = divide_and_conquer.measurePoints
        using_divide = True
    measurement = ClassicalRegister(b_register.size + 1, name='measurement')

    circuit.add_register(ancilla_register)
    circuit.add_register(c_register)
    circuit.add_register(measurement)

    Umatrix, eigs = u_matrix(A, t=t, debug=print_circuit)
    CU = cu_gate(Umatrix)  # DeprecationWarning!
    CU_Inverse = cu_gate_inverse(Umatrix)  # DeprecationWarning!
    circuit.barrier()

    # Uncomment the two lines below to measure the b vector to see if it is loaded correctly
    # for i in range(b_register.size):
    #     circuit.measure(b_register[i], measurement[i + 1])

    # ---------QPE------------
    circuit.h(c_register)

    for k in range(c_register.size):
        for i in range(np.power(2, k)):
            circuit.append(CU, [c_register[k], *b_register])
    # IQFT
    inv_qft = create_qft_inverse(c_register.size, print_circuit)
    circuit.append(inv_qft, c_register)

    # ---------RY-------------
    eigTilde = np.abs((eigs * t / (2 * np.pi)) * 2 ** c_register.size)
    condition_number = np.abs(np.max(eigs)) / np.abs(np.min(eigs))  # Serching... NOT GOOD
    if print_circuit:
        print("A", A)
        print('Encoded eigen values', eigTilde)
        print("Condition number:", condition_number)
        print("Condition number:", np.linalg.cond(A))
    C = 1 / condition_number

    for i in range(c_register.size):
        # circuit.cry(rY_roation(eigTilde[i], C), cRegister[i], ancillaRegister)
        circuit.cry(ry_rotation(eigTilde[c_register.size - 1 - i], C), c_register[i], ancilla_register)
    circuit.measure(ancilla_register, measurement[0])

    # --------IQPE-------------
    # QFT
    _qft = create_qft(c_register.size, print_circuit)
    circuit.append(_qft, c_register)
    for k in range(c_register.size):
        for i in range(np.power(2, c_register.size - 1 - k)):
            circuit.append(CU_Inverse, [c_register[c_register.size - 1 - k], *b_register])

    circuit.barrier()
    circuit.h(c_register)

    # Measurements-----------------
    for i in range(b_register.size):
        circuit.measure(b_register[i], measurement[i + 1])

    # HHL finished!-------------------------
    if print_circuit:
        try:
            circuit.draw(output='mpl').show()
        except:
            print(circuit.draw(output='text'))

    return circuit
