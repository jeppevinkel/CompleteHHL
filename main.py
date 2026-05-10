from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from tests import Tests, Test
import csv

r""" Guide for using this program:
We have implemented a Tests class that can run a testxx, defined in the class itself. A test consists of an A matrix and
b vector. Feel free to add additional tests in the class.
The runTest(testxx) method attempts to solve for x in Ax=b, using OUR implementation of the HHL algorithm in hhl.py.
The runQiskitTest(testxx) method attempts to solve for x in Ax=b, using Qiskit's implementation of the HHL algorithm
in https://qiskit.org/documentation/stubs/qiskit.algorithms.HHL.html?highlight=hhl#qiskit.algorithms.HHL.
"""


def main():
    start = datetime.now()
    print("Start: ", start, '\n')
    testClass = Tests(debug=True)
    test = testClass.tests[0]

    for offset in range(0, 4):
        try:
            print(test.name, '\nnc_offset: ', offset)
            q_tests = []

            print("A", test.A)
            print("b", test.b)
#
            q_res = testClass.run_qiskit_test(test, offset)
            #q_tests.append(q_res)
            print("Result:", q_res)
            print("Result^2:", np.power(q_res, 2))
            print("sqrt(Result):", np.sqrt(q_res))

            n = round(np.sqrt(len(np.sqrt(q_res))))
            u = np.sqrt(q_res).reshape(n, n)
            u = np.pad(u, (1, 1))  # add boundary
            xx, yy = np.meshgrid(np.linspace(0, 1, n + 2), np.linspace(0, 1, n + 2))
            ax = plt.axes(projection='3d')
            ax.plot_surface(xx, yy, u, cmap='viridis')
            ax.set_title(f'N = {n + 1}, nc_offset = {offset}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('u')
            plt.show()
        except:
            print("Error")

    end = datetime.now()
    print("\nExecution time: ", end - start)
    print("End: ", end)


if __name__ == '__main__':
    main()
