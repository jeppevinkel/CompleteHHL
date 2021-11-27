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
    # t = np.pi
    a = 1
    b = 1 / 2
    ancillaRegister = QuantumRegister(1, name='ancilla')
    cRegister = QuantumRegister(2, name='clock')
    bRegister = QuantumRegister(1, name='b')
    measurement = ClassicalRegister(2, name='measurement')
    circuit = QuantumCircuit(ancillaRegister, cRegister, bRegister, measurement)

    # circuit.x(bRegister)

    # Quantum Phase Estimation #
    # H
    circuit.h(cRegister)

    # circuit.barrier()
    # e^(iAt)

    circuit.cu(np.pi, 3 * np.pi / 2, 5 * np.pi / 2, 0, cRegister[0], bRegister)
    circuit.cu(np.pi, 3 * np.pi / 2, 5 * np.pi / 2, 0, cRegister[1], bRegister)
    circuit.cu(np.pi, 3 * np.pi / 2, 5 * np.pi / 2, 0, cRegister[1], bRegister)
    # circuit.cu(-2*np.pi, 0, 0, 0, cRegister[0], bRegister)
    # circuit.cu(3*np.pi, (5*np.pi)/2, (3*np.pi)/2, 0, cRegister[1], bRegister)

    # circuit.barrier()
    # IQFT
    inv_qft = create_qft_inverse(cRegister.size)
    circuit.append(inv_qft, cRegister)

    # circuit.barrier()
    # RY
    circuit.cry(2 * np.arcsin(1 / 3), cRegister[0], ancillaRegister)
    circuit.cry(2 * np.arcsin(1), cRegister[1], ancillaRegister)
    # circuit.cry(np.pi, cRegister[0], ancillaRegister)
    # circuit.cry(np.pi/3, cRegister[1], ancillaRegister)

    # circuit.barrier()
    # measure ancilla
    circuit.measure(ancillaRegister, measurement[0])

    # circuit.barrier()
    # Inverse Quantum Phase Estimation #
    # QFT
    _qft = create_qft(cRegister.size)
    circuit.append(_qft, cRegister)

    # circuit.barrier()
    # e^(-iAt)
    circuit.cu(np.pi, 3 * np.pi / 2, 5 * np.pi / 2, 0, cRegister[1], bRegister)
    circuit.cu(np.pi, 3 * np.pi / 2, 5 * np.pi / 2, 0, cRegister[1], bRegister)
    circuit.cu(np.pi, 3 * np.pi / 2, 5 * np.pi / 2, 0, cRegister[0], bRegister)
    # circuit.cu(5 * np.pi, (5 * np.pi) / 2, (3 * np.pi) / 2, 0, cRegister[1], bRegister)
    # circuit.cu(-2*np.pi, 0, 0, 0, cRegister[0], bRegister)

    # circuit.barrier()
    # H
    circuit.h(cRegister)

    # qft_dagger(circuit, cRegister)
    # circuit.barrier()
    circuit.measure(bRegister, measurement[1])

    circuit.draw(output='mpl').show()

    aer_sim = Aer.get_backend('aer_simulator')
    shots = 2048
    t_circuit = transpile(circuit, aer_sim)
    job = aer_sim.run(t_circuit, shots=shots)
    job_monitor(job)
    results = job.result()
    answer = results.get_counts(circuit)

    plot_histogram(answer).show()
