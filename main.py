import numpy as np
from qiskit import IBMQ, Aer, transpile, assemble
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library.standard_gates import RYGate
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.circuit import Qubit
from qiskit.extensions import UnitaryGate
import unitaryDecomposition
import divideAndConquer
import hhl


def main():
    A = np.array([[1, 1 / 2],
                  [1 / 2, 1]])
    b = np.array([[1], [0]])

    A2 = np.array([[4, -1, -1, -1],
                   [-1, 4, -1, -1],
                   [-1, -1, -1, -1],
                   [-1, -1, -1, 4]])

    b2 = np.array([[4], [1], [9], [0.6]])

    circuit = hhl.hhl(A, b, np.pi, printCircuit=True)

    # -----Simulation------------
    backend = Aer.get_backend('aer_simulator')
    t_circuit = transpile(circuit, backend)
    result = backend.run(t_circuit, shots=2048).result()
    counts = result.get_counts()
    filteredCounts = dict()
    for key, value in counts.items():
        if key[len(key)-1] == '1':
            filteredCounts[key] = value
    print(counts)
    print(filteredCounts)
    if len(filteredCounts):
        plot_histogram(filteredCounts).show()
    plot_histogram(counts).show()


if __name__ == '__main__':
    main()
