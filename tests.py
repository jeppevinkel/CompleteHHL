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

    test0 = Test(np.array([[10, -8.881784e-16], [8.881784e-16, 10]]), np.array([[1], [1]]), 'test 0')
    test1 = Test(np.array([[7.549020e+00, 1.470588e+00], [1.470588e+00, 9.117647e+00]]),
                 np.array([[1], [1.173913e+00]]), 'test 1')
    test2 = Test(np.array([[5.915033e+00, 2.450980e+00], [2.450980e+00, 8.529412e+00]]),
                 np.array([[1], [1.312500e+00]]), 'test 2')
    test3 = Test(np.array([[4.825708e+00, 3.104575e+00], [3.104575e+00, 8.137255e+00]]),
                 np.array([[1], [1.417582e+00]]), 'test 3')
    test4 = Test(np.array([[4.099492e+00, 3.540305e+00], [3.540305e+00, 7.875817e+00]]),
                 np.array([[1], [1.494297e+00]]), 'test 4')
    test5 = Test(np.array([[3.615347e+00, 3.830792e+00], [3.830792e+00, 7.701525e+00]]),
                 np.array([[1], [1.548765e+00]]), 'test 5')
    test6 = Test(np.array([[3.292585e+00, 4.024449e+00], [4.024449e+00, 7.585330e+00]]),
                 np.array([[1], [1.586678e+00]]), 'test 6')
    test7 = Test(np.array([[3.077409e+00, 4.153554e+00], [4.153554e+00, 7.507867e+00]]),
                 np.array([[1], [1.612706e+00]]), 'test 7')
    test8 = Test(np.array([[2.933959e+00, 4.239625e+00], [4.239625e+00, 7.456225e+00]]),
                 np.array([[1], [1.630405e+00]]), 'test 8')
    test9 = Test(np.array([[2.838326e+00, 4.297005e+00], [4.297005e+00, 7.421797e+00]]),
                 np.array([[1], [1.642363e+00]]), 'test 9')
    test10 = Test(np.array([[2.774570e+00, 4.335258e+00], [4.335258e+00, 7.398845e+00]]),
                  np.array([[1], [1.650406e+00]]), 'test 10')

    tests: list = [
        test0,
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

        # test12,
        # test13,
        # test14,
        # test15,
        # test16,
        # test17,
        # test18,
        # test19,
        # test20,
        # test21,
        # test22,
        # test23,
        # test24,
        # test25,
        # test26,
        # test27,
        # test28,
        # test29,
        # test30,
        # test31,
        # test32,
        # test33,
        # test34,
        # test35,
        # test36,
        # test37,
        # test38,
        # test39,
        # test40,
        # test41,
        # test42,
        # test43,
        # test44,
        # test45,
        # test46,
        # test47,
        # test48,
        # test49,
        # test50
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
