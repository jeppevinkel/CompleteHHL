import numpy as np
from qiskit import IBMQ, Aer, transpile, assemble
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library.standard_gates import RYGate
from qiskit.visualization import plot_histogram


def level(index):
    return int(np.floor(np.log2(index + 1)))


def left(n):
    return 2 * n + 1


def right(n):
    return 2 * n + 2


def gen_angles(x):
    if len(x) > 1:
        new_length = int(len(x) / 2)
        angles = [0 for i in range(new_length)]
        new_x = [0 for i in range(new_length)]

        for k in range(0, new_length):
            new_x[k] = np.sqrt(np.power(x[2 * k], 2) + np.power(x[2 * k + 1], 2))
        inner_angles = gen_angles(new_x)

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


def gen_angles_z(x):
    if len(x) > 1:
        new_length = int(len(x) / 2)
        angles_z = [0 for i in range(new_length)]
        new_x = [0 for i in range(new_length)]

        for k in range(new_length):
            new_x[k] = (x[2 * k] + x[2 * k + 1]) / 2
        inner_angles_z = gen_angles_z(new_x)

        for k in range(new_length):
            angles_z[k] = x[2 * k + 1] - x[2 * k]

        if inner_angles_z is not None:
            angles_z = inner_angles_z + angles_z
        return angles_z


class DivideAndConquer:
    N: int
    circuit: QuantumCircuit
    register: QuantumRegister
    measurePoints = np.array([], dtype=int)

    def __init__(self, circuit: QuantumCircuit):
        self.circuit = circuit

    def divide_and_conquer_complex(self, angles: list, angles_z: list):
        self.N = len(angles) + 1
        N = self.N
        bRegister = QuantumRegister(N-1, 'b')
        self.register = bRegister
        self.circuit.add_register(bRegister)
        for k in range(N - 1):
            self.circuit.ry(bRegister[angles[k]], k)
        for k in range(N - 1):
            self.circuit.rz(bRegister[angles_z[k]], k)

        actual = N - 2
        while actual >= 0:
            left_index = left(actual)
            right_index = right(actual)
            while right_index < N - 1:
                self.circuit.cswap(bRegister[actual], bRegister[left_index], bRegister[right_index])
                left_index = left(left_index)
                right_index = left(right_index)
            actual = actual - 1

        actual = 0
        while actual < (N - 1) / 2:
            self.measurePoints = np.append(self.measurePoints, actual)
            actual = left(actual)
        self.measurePoints = self.measurePoints[::-1]

        return bRegister

    def divide_and_conquer(self, angles: list):
        self.N = len(angles) + 1
        N = self.N
        bRegister = QuantumRegister(N-1, 'b')
        self.register = bRegister
        self.circuit.add_register(bRegister)
        for k in range(N - 1):
            self.circuit.ry(angles[k], bRegister[k])

        actual = N - 2
        while actual >= 0:
            left_index = left(actual)
            right_index = right(actual)
            while right_index < N - 1:
                self.circuit.cswap(bRegister[actual], bRegister[left_index], bRegister[right_index])
                left_index = left(left_index)
                right_index = left(right_index)
            actual = actual - 1

        actual = 0
        while actual < (N - 1) / 2:
            self.measurePoints = np.append(self.measurePoints, actual)
            actual = left(actual)
        self.measurePoints = self.measurePoints[::-1]

        return bRegister

    def gen_circuit(self, angles):
        qbits = int(np.log2(len(angles) + 1))
        circuit = QuantumCircuit(qbits)
        for k in range(0, len(angles)):
            j = level(k)
            if j == 0:
                circuit.ry(k, angles[j])
            else:
                print("level: " + str(j))
                myGate = RYGate(angles[k]).control(j)
                circuit.append(myGate, range(0, j + 1))

        circuit.measure_all()
        return circuit

    def loadB(self, vector: np.ndarray):
        assert vector.size % 2 == 0

        angles = gen_angles(np.abs(vector))
        register = self.divide_and_conquer(angles)
        self.circuit.draw(output='mpl').show()
        return register

    def measureB(self, measure_register: ClassicalRegister):
        for i in range(self.measurePoints.size):
            self.circuit.measure(self.measurePoints[i], measure_register[i+1])
