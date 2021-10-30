import numpy as np
from qiskit import IBMQ, Aer, transpile, assemble
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library.standard_gates import RYGate
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.circuit import Qubit


def qft(qc: QuantumCircuit, qr: QuantumRegister):
    n = qr.size
    for q in range(qr.size):
        qc.h(q)
        r = 2
        i = q+1
        while r <= n-q:
            qc.cp(np.pi/2**(r-1), q, i)
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
    return qc


# def qft_dagger(qc, qr: QuantumRegister):
#     """n-qubit QFTdagger the first n qubits in circ"""
#     # Don't forget the Swaps!
#     for qubit in range(qr.size//2):
#         qc.swap(qr[qubit], qr[qr.size-qubit-1])
#     for j in range(qr.size):
#         for m in range(j):
#             qc.cp(-np.pi/float(2**(j-m)), qr[m], qr[j])
#         qc.h(qr[j])


if __name__ == '__main__':
    t = np.pi
    a = 1
    b = 1/2
    ancillaRegister = QuantumRegister(1, name='ancilla')
    cRegister = QuantumRegister(2, name='clock')
    bRegister = QuantumRegister(1, name='b')
    measurement = ClassicalRegister(1, name='measurement')
    circuit = QuantumCircuit(ancillaRegister, cRegister, bRegister, measurement)

    # Quantum Phase Estimation #
    # H
    circuit.h(cRegister)

    circuit.barrier()
    # e^(iAt)

    # circuit.p(a*t, cRegister[0])
    # circuit.p(a*t*2, cRegister[1])
    # circuit.u(b*t, -np.pi/2, np.pi/2, bRegister)
    #
    # params = b * t
    # circuit.p(np.pi/2, bRegister)
    # circuit.cx(cRegister[0], bRegister)
    # circuit.ry(params, bRegister)
    # circuit.cx(cRegister[0], bRegister)
    # circuit.ry(-params, bRegister)
    # circuit.p(3*np.pi/2, bRegister)
    #
    # params = b * t * 2
    # circuit.p(np.pi/2, bRegister)
    # circuit.cx(cRegister[1], bRegister)
    # circuit.ry(params, bRegister)
    # circuit.cx(cRegister[1], bRegister)
    # circuit.ry(-params, bRegister)
    # circuit.p(3*np.pi/2, bRegister)

    circuit.barrier()
    # IQFT
    inv_qft = create_qft_inverse(cRegister.size)
    circuit.append(inv_qft, cRegister)

    circuit.barrier()
    # RY
    circuit.cry(2*np.arcsin(1/3), cRegister[0], ancillaRegister)
    circuit.cry(2*np.arcsin(1), cRegister[1], ancillaRegister)

    circuit.barrier()
    # Inverse Quantum Phase Estimation #
    # QFT
    _qft = create_qft(cRegister.size)
    circuit.append(_qft, cRegister)

    circuit.barrier()
    # e^(iAt)

    # circuit.h(cRegister[1])
    # circuit.rz(-np.pi/4, cRegister[1])
    # circuit.cx(cRegister[0], cRegister[1])
    # circuit.rz(np.pi/4, cRegister[1])
    # circuit.cx(cRegister[0], cRegister[1])
    # circuit.rz(-np.pi/4, cRegister[0])
    # circuit.h(cRegister[0])
    #
    # t1 = (-np.pi + np.pi / 3 - 2 * np.arcsin(1 / 3)) / 4
    # t2 = (-np.pi - np.pi / 3 + 2 * np.arcsin(1 / 3)) / 4
    # t3 = (np.pi - np.pi / 3 - 2 * np.arcsin(1 / 3)) / 4
    # t4 = (np.pi + np.pi / 3 + 2 * np.arcsin(1 / 3)) / 4
    #
    # circuit.cx(cRegister[1], ancillaRegister[0])
    # circuit.ry(t1, ancillaRegister[0])
    # circuit.cx(cRegister[0], ancillaRegister[0])
    # circuit.ry(t2, ancillaRegister[0])
    # circuit.cx(cRegister[1], ancillaRegister[0])
    # circuit.ry(t3, ancillaRegister[0])
    # circuit.cx(cRegister[0], ancillaRegister[0])
    # circuit.ry(t4, ancillaRegister[0])
    # circuit.measure_all()

    # qft_dagger(circuit, cRegister)
    circuit.barrier()
    circuit.measure(ancillaRegister, measurement[0])

    circuit.draw(output='mpl').show()

    aer_sim = Aer.get_backend('aer_simulator')
    shots = 2048
    t_circuit = transpile(circuit, aer_sim)
    job = aer_sim.run(t_circuit, shots=shots)
    job_monitor(job)
    results = job.result()
    answer = results.get_counts(circuit)

    plot_histogram(answer).show()
