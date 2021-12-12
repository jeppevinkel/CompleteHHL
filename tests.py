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
    test12 = Test(np.array([[5.416667e+00, 4.583333e+00], [4.583333e+00, 5.416667e+00]]), np.array([[1], [1]]),
                  'test 12')
    test13 = Test(np.array([[5.384615e+00, 4.615385e+00], [4.615385e+00, 5.384615e+00]]), np.array([[1], [1]]),
                  'test 13')
    test14 = Test(np.array([[5.357143e+00, 4.642857e+00], [4.642857e+00, 5.357143e+00]]), np.array([[1], [1]]),
                  'test 14')
    test15 = Test(np.array([[5.333333e+00, 4.666667e+00], [4.666667e+00, 5.333333e+00]]), np.array([[1], [1]]),
                  'test 15')
    test16 = Test(np.array([[5.312500e+00, 4.687500e+00], [4.687500e+00, 5.312500e+00]]), np.array([[1], [1]]),
                  'test 16')
    test17 = Test(np.array([[5.294118e+00, 4.705882e+00], [4.705882e+00, 5.294118e+00]]), np.array([[1], [1]]),
                  'test 17')
    test18 = Test(np.array([[5.277778e+00, 4.722222e+00], [4.722222e+00, 5.277778e+00]]), np.array([[1], [1]]),
                  'test 18')
    test19 = Test(np.array([[5.263158e+00, 4.736842e+00], [4.736842e+00, 5.263158e+00]]), np.array([[1], [1]]),
                  'test 19')
    test20 = Test(np.array([[5.250000e+00, 4.750000e+00], [4.750000e+00, 5.250000e+00]]), np.array([[1], [1]]),
                  'test 20')
    test21 = Test(np.array([[5.238095e+00, 4.761905e+00], [4.761905e+00, 5.238095e+00]]), np.array([[1], [1]]),
                  'test 21')
    test22 = Test(np.array([[5.227273e+00, 4.772727e+00], [4.772727e+00, 5.227273e+00]]), np.array([[1], [1]]),
                  'test 22')
    test23 = Test(np.array([[5.217391e+00, 4.782609e+00], [4.782609e+00, 5.217391e+00]]), np.array([[1], [1]]),
                  'test 23')
    test24 = Test(np.array([[5.208333e+00, 4.791667e+00], [4.791667e+00, 5.208333e+00]]), np.array([[1], [1]]),
                  'test 24')
    test25 = Test(np.array([[5.200000e+00, 4.800000e+00], [4.800000e+00, 5.200000e+00]]), np.array([[1], [1]]),
                  'test 25')
    test26 = Test(np.array([[5.192308e+00, 4.807692e+00], [4.807692e+00, 5.192308e+00]]), np.array([[1], [1]]),
                  'test 26')
    test27 = Test(np.array([[5.185185e+00, 4.814815e+00], [4.814815e+00, 5.185185e+00]]), np.array([[1], [1]]),
                  'test 27')
    test28 = Test(np.array([[5.178571e+00, 4.821429e+00], [4.821429e+00, 5.178571e+00]]), np.array([[1], [1]]),
                  'test 28')
    test29 = Test(np.array([[5.172414e+00, 4.827586e+00], [4.827586e+00, 5.172414e+00]]), np.array([[1], [1]]),
                  'test 29')
    test30 = Test(np.array([[5.166667e+00, 4.833333e+00], [4.833333e+00, 5.166667e+00]]), np.array([[1], [1]]),
                  'test 30')
    test31 = Test(np.array([[5.161290e+00, 4.838710e+00], [4.838710e+00, 5.161290e+00]]), np.array([[1], [1]]),
                  'test 31')
    test32 = Test(np.array([[5.156250e+00, 4.843750e+00], [4.843750e+00, 5.156250e+00]]), np.array([[1], [1]]),
                  'test 32')
    test33 = Test(np.array([[5.151515e+00, 4.848485e+00], [4.848485e+00, 5.151515e+00]]), np.array([[1], [1]]),
                  'test 33')
    test34 = Test(np.array([[5.147059e+00, 4.852941e+00], [4.852941e+00, 5.147059e+00]]), np.array([[1], [1]]),
                  'test 34')
    test35 = Test(np.array([[5.142857e+00, 4.857143e+00], [4.857143e+00, 5.142857e+00]]), np.array([[1], [1]]),
                  'test 35')
    test36 = Test(np.array([[5.138889e+00, 4.861111e+00], [4.861111e+00, 5.138889e+00]]), np.array([[1], [1]]),
                  'test 36')
    test37 = Test(np.array([[5.135135e+00, 4.864865e+00], [4.864865e+00, 5.135135e+00]]), np.array([[1], [1]]),
                  'test 37')
    test38 = Test(np.array([[5.131579e+00, 4.868421e+00], [4.868421e+00, 5.131579e+00]]), np.array([[1], [1]]),
                  'test 38')
    test39 = Test(np.array([[5.128205e+00, 4.871795e+00], [4.871795e+00, 5.128205e+00]]), np.array([[1], [1]]),
                  'test 39')
    test40 = Test(np.array([[5.125000e+00, 4.875000e+00], [4.875000e+00, 5.125000e+00]]), np.array([[1], [1]]),
                  'test 40')
    test41 = Test(np.array([[5.121951e+00, 4.878049e+00], [4.878049e+00, 5.121951e+00]]), np.array([[1], [1]]),
                  'test 41')
    test42 = Test(np.array([[5.119048e+00, 4.880952e+00], [4.880952e+00, 5.119048e+00]]), np.array([[1], [1]]),
                  'test 42')
    test43 = Test(np.array([[5.116279e+00, 4.883721e+00], [4.883721e+00, 5.116279e+00]]), np.array([[1], [1]]),
                  'test 43')
    test44 = Test(np.array([[5.113636e+00, 4.886364e+00], [4.886364e+00, 5.113636e+00]]), np.array([[1], [1]]),
                  'test 44')
    test45 = Test(np.array([[5.111111e+00, 4.888889e+00], [4.888889e+00, 5.111111e+00]]), np.array([[1], [1]]),
                  'test 45')
    test46 = Test(np.array([[5.108696e+00, 4.891304e+00], [4.891304e+00, 5.108696e+00]]), np.array([[1], [1]]),
                  'test 46')
    test47 = Test(np.array([[5.106383e+00, 4.893617e+00], [4.893617e+00, 5.106383e+00]]), np.array([[1], [1]]),
                  'test 47')
    test48 = Test(np.array([[5.104167e+00, 4.895833e+00], [4.895833e+00, 5.104167e+00]]), np.array([[1], [1]]),
                  'test 48')
    test49 = Test(np.array([[5.102041e+00, 4.897959e+00], [4.897959e+00, 5.102041e+00]]), np.array([[1], [1]]),
                  'test 49')
    test50 = Test(np.array([[5.100000e+00, 4.900000e+00], [4.900000e+00, 5.100000e+00]]), np.array([[1], [1]]),
                  'test 50')

    tests: list = [
        test1,
        test2,
        test3,
        test4,
        test5,
        test6,
        test7,
        test8,
        test9,
        test10,
        test11,
        test12,
        test13,
        test14,
        test15,
        test16,
        test17,
        test18,
        test19,
        test20,
        test21,
        test22,
        test23,
        test24,
        test25,
        test26,
        test27,
        test28,
        test29,
        test30,
        test31,
        test32,
        test33,
        test34,
        test35,
        test36,
        test37,
        test38,
        test39,
        test40,
        test41,
        test42,
        test43,
        test44,
        test45,
        test46,
        test47,
        test48,
        test49,
        test50
    ]

    def __init__(self, debug=False):
        self.debug = debug

    # runs a test defined by a Test class using our circuit implementation
    def run_test(self, test: Test):
        circuit = hhl.hhl(test.A, test.b, t=np.pi, print_circuit=self.debug)
        self.run_simulation(circuit, test.name, 'our')

    # runs a test defined by a Test class using the implementation built into Qiskit
    def run_qiskit_test(self, test: Test):
        circuit = return_qiskit_circuit(test.A, test.b)
        if self.debug:
            circuit.draw(output='mpl').show()
        return self.run_simulation(circuit, test.name, 'qiskit')

    @staticmethod
    def run_classical_test(test: Test):
        x = lu.lu_solve(test.A, test.b)
        print('Classical solution of', test.name)
        print("LU decomp. solution:\n", x)
        print("\nExpected filtered measurement result (according to LU decomp.):\n", np.power(x/np.linalg.norm(x), 2))
        return np.power(x/np.linalg.norm(x), 2)

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
        # print('Counts', counts)
        # print('Filtered counts', filtered_counts)
        if len(filtered_counts):
            # plot_histogram(filtered_counts,
            #                title='filtered ' + test_name + ', ' + implementation + ' implementation').show()
            results = np.array([value for (key, value) in sorted(filtered_counts.items())])
            # results = np.array(list(filtered_counts.values()))
            # print('Result:', results/np.sum(results))
            return results/np.sum(results)
        else:
            # print("ANCILLA BIT NEVER 1")
            return np.array([np.NaN, np.NaN])
        # plot_histogram(counts, title='unfiltered ' + test_name + ', ' + implementation + ' implementation').show()
