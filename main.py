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

    for i, test in np.ndenumerate(testClass.tests):
        print(test.name)
        q_tests = []

        print("A", test.A)
        print("b", test.b)
#
        q_res = testClass.run_test(test)
        #q_tests.append(q_res)
        print("Result:", q_res)
        print("Result^2:", np.power(q_res, 2))
        print("sqrt(Result):", np.sqrt(q_res))

    end = datetime.now()
    print("\nExecution time: ", end - start)
    print("End: ", end)


if __name__ == '__main__':
    main()
