from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

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
    with open('test_data.csv', mode='w', newline='') as test_file:
        test_writer = csv.writer(test_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL, dialect='excel')
        test_writer.writerow(['Condition number', 'Error', 'NaN count'])

        rows: list = []
        error: list = []
        nan: list = []

        testClass = Tests(debug=False)

        for i, test in np.ndenumerate(testClass.tests):
            print('Condition number:', i[0]+1)

            q_tests = []
            d = 0
            num_nan = 0

            for _ in range(10):
                # print('\nTest using the Qiskit implementation')
                q_res = testClass.run_qiskit_test(test)
                q_tests.append(q_res)

            # print('\nTest using the classical implementation')
            # c_res = testClass.run_classical_test(test)

            # print("q_res", q_res)
            # print("c_res", c_res)

            for q_res in q_tests:
                if q_res.size == 2 and not np.isnan(q_res[0]) and not np.isnan(q_res[1]):
                    d = d + np.sqrt(np.power((q_res[0] - 0.5), 2) + np.power((q_res[1] - 0.5), 2))
                else:
                    # d = d + np.NaN
                    # d = d + 1
                    num_nan = num_nan + 1
            size = (len(q_tests) - num_nan)
            if size > 0:
                d = d/(len(q_tests) - num_nan)
            else:
                d = np.NaN

            rows.append([i[0]+1, d, num_nan])
            error.append(d)
            nan.append(num_nan)


        test_writer.writerows(rows)

        x_size = [*range(1, len(testClass.tests)+1)]
        fig, x1 = plt.subplots()
        x2 = x1.twinx()  # instantiate a second axes that shares the same x-axis

        x1.plot(x_size, error, color='black', markerfacecolor='black', linestyle=':', marker='s')
        x1.set_xlabel('Condition number')
        x1.set_ylabel('Average error')

        color = 'grey'
        x2.set_ylabel('Times the measurement is invalid', color=color)  # we already handled the x-label with ax1
        x2.bar([*range(1, len(testClass.tests)+1)], nan, color=color, alpha=0.150)
        x2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

        end = datetime.now()
        print("\nExecution time: ", end - start)
        print("End: ", end)


if __name__ == '__main__':
    main()
