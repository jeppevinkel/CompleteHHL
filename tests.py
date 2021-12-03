import numpy as np
from qiskit import Aer, transpile, QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskitImplementation import return_qiskit_circuit
import lu

import hhl


def filip_data():
    data = np.loadtxt('Filip.dat', skiprows=60)
    assert data.shape == (82, 2)

    y_filip, x_filip = data.T
    a = x_filip.reshape((-1, 1)) ** np.arange(11)
    b = y_filip[..., None]
    return Test(a, b, 'Filip')


class Test:
    A: np.ndarray
    b: np.ndarray
    name: str

    def __init__(self, a: np.ndarray, b: np.ndarray, name: str = 'Test'):
        self.A = a
        self.b = b
        self.name = name


class Tests:
    debug: bool

    test01 = Test(np.array([[1, 1 / 2], [1 / 2, 1]]), np.array([[1], [0]]), 'Test01')
    test02 = Test(np.array([[4, 1, 1, 1], [1, 4, 1, 1], [1, 1, 4, 1], [1, 1, 1, 4]]), np.array([[4], [1], [9], [3]]),
                  'Test02')
    test03 = Test(np.array([[7, 9], [1, 3]]), np.array([[4], [6]]), 'Test03')
    test04 = Test(np.array([[2, 7, 8], [4, 5, 2], [3, 1, 6]]), np.array([[4], [6], [23]]), 'Test04')
    test_filip = filip_data()

    def __init__(self, debug=False):
        self.debug = debug

    # runs a test defined by a Test class using our circuit implementation
    def run_test(self, test: Test):
        circuit = hhl.hhl(test.A, test.b, t=np.pi, print_circuit=self.debug)
        self.run_simulation(circuit, test.name, 'our')

    # runs a test defined by a Test class using the implementation built into Qiskit
    def run_qiskit_test(self, test: Test):
        circuit = return_qiskit_circuit(test.A, test.b)
        self.run_simulation(circuit, test.name, 'qiskit')

    @staticmethod
    def run_classical_test(test: Test):
        x = lu.lu_solve(test.A, test.b)
        print('Classical solution of', test.name)
        print("LU decomp. solution:\n", x)
        print("\nExpected filtered measurement result (according to LU decomp.):\n", np.power(x/np.linalg.norm(x), 2))

    # runs a simulation using a HHL circuit assuming LSB is the ancilla bit
    @staticmethod
    def run_simulation(circuit: QuantumCircuit, test_name: str, implementation: str):
        backend = Aer.get_backend('aer_simulator')
        t_circuit = transpile(circuit, backend)
        result = backend.run(t_circuit, shots=4096).result()
        counts = result.get_counts()
        filtered_counts = dict()
        for key, value in counts.items():
            if key[len(key) - 1] == '1':
                filtered_counts[key] = value
        print('Counts', counts)
        print('Filtered counts', filtered_counts)
        if len(filtered_counts):
            plot_histogram(filtered_counts,
                           title='filtered ' + test_name + ', ' + implementation + ' implementation').show()
            results = np.array(list(filtered_counts.values()))
            print('Result:', results/np.sum(results))
        else:
            print("ANCILLA BIT NEVER 1")
        plot_histogram(counts, title='unfiltered ' + test_name + ', ' + implementation + ' implementation').show()
