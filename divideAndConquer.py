import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.standard_gates import RYGate

r""" This code loads a vector b into the quantum circuit. The algorithm implemented here is from:
https://www.nature.com/articles/s41598-021-85474-1.pdf (Last visited: 03/12/2021)
"""


class DivideAndConquer:
    N: int
    circuit: QuantumCircuit
    register: QuantumRegister
    measurePoints = np.array([], dtype=int)

    def __init__(self, circuit: QuantumCircuit):
        self.circuit = circuit

    def divide_and_conquer(self, angles: list, angles_z: list = None):
        self.N = len(angles) + 1
        n = self.N
        b_register = QuantumRegister(n - 1, 'b')
        self.register = b_register
        self.circuit.add_register(b_register)
        for k in range(n - 1):
            self.circuit.ry(angles[k], b_register[k])
        for k in range(n - 1):
            self.circuit.rz(angles_z[k], b_register[k])

        actual = n - 2
        while actual >= 0:
            left_index = DivideAndConquer.left(actual)
            right_index = DivideAndConquer.right(actual)
            while right_index < n - 1:
                self.circuit.cswap(b_register[actual], b_register[left_index], b_register[right_index])
                left_index = DivideAndConquer.left(left_index)
                right_index = DivideAndConquer.left(right_index)
            actual = actual - 1

        actual = 0
        while actual < (n - 1) / 2:
            self.measurePoints = np.append(self.measurePoints, actual)
            actual = DivideAndConquer.left(actual)
        self.measurePoints = self.measurePoints[::-1]

        return b_register

    @staticmethod
    def gen_circuit(angles):
        qbits = int(np.log2(len(angles) + 1))
        circuit = QuantumCircuit(qbits)
        for k in range(0, len(angles)):
            j = DivideAndConquer.level(k)
            if j == 0:
                circuit.ry(k, angles[j])
            else:
                print("level: " + str(j))
                my_gate = RYGate(angles[k]).control(j)
                circuit.append(my_gate, range(0, j + 1))

        circuit.measure_all()
        return circuit

    @staticmethod
    def level(index: int):
        return int(np.floor(np.log2(index + 1)))

    @staticmethod
    def left(n):
        return 2 * n + 1

    @staticmethod
    def right(n):
        return 2 * n + 2

    @staticmethod
    def gen_angles(x):
        if len(x) > 1:
            new_length = int(len(x) / 2)
            angles = [0 for _ in range(new_length)]
            new_x = [0 for _ in range(new_length)]

            for k in range(0, new_length):
                new_x[k] = np.sqrt(np.power(x[2 * k], 2) + np.power(x[2 * k + 1], 2))
            inner_angles = DivideAndConquer.gen_angles(new_x)

            for k in range(0, len(new_x)):
                if new_x[k] != 0:
                    if x[2 * k] > 0:
                        angles[k] = 2 * np.arcsin((x[2 * k + 1]) / (new_x[k]))
                    else:
                        angles[k] = 2 * np.pi - 2 * np.arcsin((x[2 * k + 1]) / (new_x[k]))
                else:
                    angles[k] = 0

            if inner_angles is not None:
                angles = inner_angles + angles
            return angles

    @staticmethod
    def gen_angles_z(x):
        if len(x) > 1:
            new_length = int(len(x) / 2)
            angles_z = [0 for _ in range(new_length)]
            new_x = [0 for _ in range(new_length)]

            for k in range(new_length):
                new_x[k] = (x[2 * k] + x[2 * k + 1]) / 2
            inner_angles_z = DivideAndConquer.gen_angles_z(new_x)

            for k in range(new_length):
                angles_z[k] = x[2 * k + 1] - x[2 * k]

            if inner_angles_z is not None:
                angles_z = inner_angles_z + angles_z
            return angles_z

    def load_b(self, vector: np.ndarray):
        assert vector.size % 2 == 0

        angles = DivideAndConquer.gen_angles(np.abs(vector))
        register = self.divide_and_conquer(angles)
        print(vector.size)
        return register
