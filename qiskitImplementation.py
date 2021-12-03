import numpy as np
from qiskit import ClassicalRegister, Aer, transpile
from qiskit.algorithms import HHL
from qiskit.visualization import plot_histogram
from unitaryDecomposition import mat_to_even_hermitian


def return_qiskit_circuit(a: np.ndarray, b: np.ndarray):
    _a, _b = mat_to_even_hermitian(a, b)

    qiskit_hhl = HHL()
    hhl_circuit = qiskit_hhl.construct_circuit(_a, _b)
    ancilla_register = hhl_circuit.qregs[-1]
    b_register = hhl_circuit.qregs[0]
    m_register = ClassicalRegister(ancilla_register.size + b_register.size)

    hhl_circuit.add_register(m_register)
    hhl_circuit.measure(ancilla_register, 0)
    hhl_circuit.measure(b_register, range(1, b_register.size + 1))
    return hhl_circuit
