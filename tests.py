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

    test1 = Test(np.array([[10, 0], [0, 10]]), np.array([[1], [1]]), 'test 1')
    test2 = Test(np.array([[7.500000e+00, 2.500000e+00], [2.500000e+00, 7.500000e+00]]), np.array([[1], [1]]), 'test 2')
    test3 = Test(np.array([[6.666667e+00, 3.333333e+00], [3.333333e+00, 6.666667e+00]]), np.array([[1], [1]]), 'test 3')
    test4 = Test(np.array([[6.250000e+00, 3.750000e+00], [3.750000e+00, 6.250000e+00]]), np.array([[1], [1]]), 'test 4')
    test5 = Test(np.array([[6, 4], [4, 6]]), np.array([[1], [1]]), 'test 5')
    test6 = Test(np.array([[5.833333e+00, 4.166667e+00], [4.166667e+00, 5.833333e+00]]), np.array([[1], [1]]), 'test 6')
    test7 = Test(np.array([[5.714286e+00, 4.285714e+00], [4.285714e+00, 5.714286e+00]]), np.array([[1], [1]]), 'test 7')
    test8 = Test(np.array([[5.625000e+00, 4.375000e+00], [4.375000e+00, 5.625000e+00]]), np.array([[1], [1]]), 'test 8')
    test9 = Test(np.array([[5.555556e+00, 4.444444e+00], [4.444444e+00, 5.555556e+00]]), np.array([[1], [1]]), 'test 9')
    test10 = Test(np.array([[5.500000e+00, 4.500000e+00], [4.500000e+00, 5.500000e+00]]), np.array([[1], [1]]),
                  'test 10')
    test11 = Test(np.array([[5.454545e+00, 4.545455e+00], [4.545455e+00, 5.454545e+00]]), np.array([[1], [1]]),
                  'test 11')

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
