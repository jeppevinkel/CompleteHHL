import numpy as np
from qiskit import ClassicalRegister, Aer, transpile
from qiskit.algorithms import HHL
from qiskit.visualization import plot_histogram
from unitaryDecomposition import MatToEvenHermitian


def return_qiskit_circuit(a: np.ndarray, b: np.ndarray):
    C, c = MatToEvenHermitian(a, b)

    qiskitsHHL = HHL()
    hhlCircuit = qiskitsHHL.construct_circuit(C, c)
    ancillaRegister = hhlCircuit.qregs[-1]
    bRegister = hhlCircuit.qregs[0]
    m_register = ClassicalRegister(ancillaRegister.size + bRegister.size)

    hhlCircuit.add_register(m_register)
    hhlCircuit.measure(ancillaRegister, 0)
    hhlCircuit.measure(bRegister, range(1, bRegister.size + 1))
    return hhlCircuit

def run_qiskit_simulation(a: np.ndarray, b: np.ndarray):
    hhlCircuit = return_qiskit_circuit(a, b)
    hhlCircuit.draw(output='mpl').show()

    backend = Aer.get_backend('aer_simulator')
    t_circuit = transpile(hhlCircuit, backend)
    result = backend.run(t_circuit, shots=8192).result()
    counts = result.get_counts()
    filteredCounts = dict()
    for key, value in counts.items():
        if key[len(key) - 1] == '1':
            filteredCounts[key] = value
    print('Counts', counts)
    print('Filtered counts', filteredCounts)
    if len(filteredCounts):
        plot_histogram(filteredCounts).show()
    else:
        print("ANCILLA BIT NEVER 1")
    plot_histogram(counts).show()
