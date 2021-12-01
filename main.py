import numpy as np
from qiskit import IBMQ, Aer, transpile, assemble
from qiskit.circuit.library.arithmetic.exact_reciprocal import ExactReciprocal
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library.standard_gates import RYGate
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.circuit import Qubit
from qiskit.algorithms.linear_solvers import HHL
from qiskit.algorithms.linear_solvers.observables import MatrixFunctional
from datetime import datetime
import hhl
from unitaryDecomposition import MatToEvenHermitian


def main():
    start = datetime.now()
    print("Start: ", start)

    A = np.array([[1, 1 / 2],
                  [1 / 2, 1]])
    b = np.array([[3.5], [5.2]])

    A2 = np.array([[4, 1, 1, 1],
                   [1, 4, 1, 1],
                   [1, 1, 4, 1],
                   [1, 1, 1, 4]])

    b2 = np.array([[4], [1], [9], [3]])

    #Loading Filip
    # filip_data = np.loadtxt('Filip.dat', skiprows=60)
    # assert filip_data.shape == (82, 2)
    #
    # y_filip, x_filip = filip_data.T
    # A = x_filip.reshape((-1, 1)) ** np.arange(11)
    # b = y_filip[..., None]
    #
    A = np.array([[7, 9],
                  [1, 3]])
    b = np.array([[4], [6]])
    #A = np.array([[2, 7, 8],
    #              [4, 5, 2],
    #              [3, 1, 6]])
    #b = np.array([[4], [6], [23]])

    H, b2 = MatToEvenHermitian(A, b)
    # b2 = np.array([[34], [54]])
    # b2 = b2 / np.linalg.norm(b)

    print("H:", H)
    print("b:", b2)

    qiskitsHHL = HHL()
    hhlCircuit = qiskitsHHL.construct_circuit(H, b2)
    ancillaRegister = hhlCircuit.qregs[-1]
    bRegister = hhlCircuit.qregs[0]
    m_register = ClassicalRegister(ancillaRegister.size + bRegister.size)

    hhlCircuit.add_register(m_register)
    hhlCircuit.measure(ancillaRegister, 0)
    hhlCircuit.measure(bRegister, range(1, bRegister.size + 1))
    hhlCircuit.draw(output='mpl').show()
    # solution = qiskitsHHL.solve(A, b)
    # print(solution)

    backend = Aer.get_backend('aer_simulator')
    t_circuit = transpile(hhlCircuit, backend)
    result = backend.run(t_circuit, shots=8192).result()
    counts = result.get_counts()
    filteredCounts = dict()
    for key, value in counts.items():
        if key[len(key) - 1] == '1':
            filteredCounts[key] = value
    print(counts)
    print(filteredCounts)
    if len(filteredCounts):
        plot_histogram(filteredCounts).show()
    else:
        print("ANCILLA BIT NEVER 1")
    plot_histogram(counts).show()

    exit()

    circuit = hhl.hhl(A, b, t=np.pi, printCircuit=True)

    # -----Simulation------------
    backend = Aer.get_backend('aer_simulator')
    t_circuit = transpile(circuit, backend)
    result = backend.run(t_circuit, shots=4096).result()
    counts = result.get_counts()
    filteredCounts = dict()
    for key, value in counts.items():
        if key[len(key)-1] == '1':
            filteredCounts[key] = value
    print(counts)
    print(filteredCounts)
    if len(filteredCounts):
        plot_histogram(filteredCounts).show()
    else:
        print("ANCILLA BIT NEVER 1")
    plot_histogram(counts).show()

    end = datetime.now()
    print("Execution time: ", end - start)
    print("End: ", end)


if __name__ == '__main__':
    main()
