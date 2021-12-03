from datetime import datetime
from tests import Tests

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

    testClass = Tests()

    print('Test using our implementation')
    testClass.runTest(testClass.test01)

    print('\nTest sing the Qiskit implementation')
    testClass.runQiskitTest(testClass.test01)

    end = datetime.now()
    print("\nExecution time: ", end - start)
    print("End: ", end)


if __name__ == '__main__':
    main()
