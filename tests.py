import numpy as np
from qiskit import Aer, transpile, QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskitImplementation import return_qiskit_circuit
import lu
import pandas as pd

import hhl

class Test:
    A: np.ndarray
    b: np.ndarray
    name: str

    def __init__(self, a: np.ndarray, b: np.ndarray, name: str = 'Test'):
        self.A = a
        self.b = b
        self.name = name


class PDE:
    A: np.ndarray
    phi: np.ndarray

    @staticmethod
    def f(x, y):
        return 1 + x + y

    def __init__(self, N):
        x0 = y0 = 0.
        x1 = y1 = 1.

        h = (x1 - x0) / N  # h is the same for y
        n = N - 1  # since u(x, y) is specified at the boundary (u=0), we have (N-1)^2 unknowns.

        # A @ u = phi
        # Ab: A banded
        Ab = np.zeros((n + 1 + n, n * n))
        self.phi = np.empty(n * n)

        # for every (col, row) in the grid of unknowns
        for col in range(n):
            x = x0 + (col + 1) * h
            for row in range(n):
                y = y0 + (row + 1) * h
                j = row * n + col  # index of unknown
                Ab[n, j] = 4
                self.phi[j] = h ** 2 * PDE.f(x, y)
                if col > 0:  # interior point to the left
                    Ab[n - 1, j] = -1
                if col < n - 1:  # interior point to the right
                    Ab[n + 1, j] = -1
                if row > 0:  # interior point below
                    Ab[0, j] = -1
                if row < n - 1:  # interior point above
                    Ab[2 * n, j] = -1

        self.A = np.zeros((Ab.shape[1], Ab.shape[1]))
        diags = int(Ab.shape[0] / 2)
        for diag in range(0, int(Ab.shape[0] / 2)):
            print("diag", diag)
            for el in range(0, Ab.shape[1] - (diags - diag)):
                self.A[el, diags - diag + el] = Ab[diag, diags - diag + el]
                self.A[(diags - diag) + el, el] = Ab[(diags * 2) - diag, el]

        for el in range(0, Ab.shape[1]):
            self.A[el, el] = Ab[diags, el]


class Tests:
    debug: bool

    test0 = np.array(PDE(4).A, PDE(4).phi, "N=4")

    tests: list = [
        test0
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
