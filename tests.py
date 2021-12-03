import numpy as np
from qiskit import Aer, transpile, QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskitImplementation import return_qiskit_circuit

import hhl


def filip_data():
    filip_data = np.loadtxt('Filip.dat', skiprows=60)
    assert filip_data.shape == (82, 2)

    y_filip, x_filip = filip_data.T
    A = x_filip.reshape((-1, 1)) ** np.arange(11)
    b = y_filip[..., None]
    return Test(A, b, 'Filip')


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

    def runTests(self):
        print("Test 01")
        self.runTest(self.test01)
        print("Test 02")
        self.runTest(self.test02)
        print("Test 03")
        self.runTest(self.test03)
        print("Test 04")
        self.runTest(self.test04)
        return 1

    # runs a test defined by a Test class using our circuit implementation
    def runTest(self, test: Test):
        circuit = hhl.hhl(test.A, test.b, t=np.pi, printCircuit=self.debug)
        self.runSimulation(circuit, test.name, 'our')

    # runs a test defined by a Test class using the implementation built into Qiskit
    def runQiskitTest(self, test: Test):
        circuit = return_qiskit_circuit(test.A, test.b)
        self.runSimulation(circuit, test.name, 'qiskit')

    # runs a simulation using a HHL circuit assuming LSB is the ancilla bit
    def runSimulation(self, circuit: QuantumCircuit, testName: str, implementation: str):
        backend = Aer.get_backend('aer_simulator')
        t_circuit = transpile(circuit, backend)
        result = backend.run(t_circuit, shots=4096).result()
        counts = result.get_counts()
        filteredCounts = dict()
        for key, value in counts.items():
            if key[len(key) - 1] == '1':
                filteredCounts[key] = value
        print('Counts', counts)
        print('Filtered counts', filteredCounts)
        if len(filteredCounts):
            plot_histogram(filteredCounts,
                           title='filtered ' + testName + ', ' + implementation + ' implementation').show()
        else:
            print("ANCILLA BIT NEVER 1")
        plot_histogram(counts, title='unfiltered ' + testName + ', ' + implementation + ' implementation').show()
