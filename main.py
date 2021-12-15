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
        test_writer.writerow(['Condition number', 'Error', 'NaN count', '', '', 'Individual test results'])

        rows: list = []
        result_rows: list = []
        error: list = []
        nan: list = []
        x_size = []

        testClass = Tests(debug=False)

        condition_base = 1.5

        for i, test in np.ndenumerate(testClass.tests):
            condition = np.linalg.cond(test.A)
            print('Condition number:', condition, ' = 1.5^' + str(i[0]))
            x_size.append(condition)

            q_tests = []
            d = 0
            num_nan = 0
            q_errors = []

            for _ in range(50):
                # print('\nTest using the Qiskit implementation')
                q_res = testClass.run_qiskit_test(test)
                q_tests.append(q_res)

            # print('\nTest using the classical implementation')
            # c_res = testClass.run_classical_test(test)

            # print("q_res", q_res)
            # print("c_res", c_res)

            for q_res in q_tests:
                if q_res.size == 2 and not np.isnan(q_res[0]) and not np.isnan(q_res[1]):
                    cur_d = np.sqrt(np.power((q_res[0] - 0.5), 2) + np.power((q_res[1] - 0.5), 2))
                    d = d + cur_d
                    q_errors.append(np.sqrt(np.power((q_res[0] - 0.5), 2) + np.power((q_res[1] - 0.5), 2)))
                    result_rows.append([condition, q_res[0], q_res[1]])
                    variance = variance + (q_res[0] - 0.5) ** 2 + (q_res[1] - 0.5) ** 2
                    # est_variance = est_variance + (cur_d - est_mu)
                else:
                    # d = d + np.NaN
                    # d = d + 1
                    num_nan = num_nan + 1
                    q_errors.append(np.NaN)

            size = (len(q_tests) - num_nan)
            if size > 0:
                d = d / (len(q_tests) - num_nan)
            else:
                d = np.NaN
            print("Average error:", d)
            expected_avg_error = 0.353553
            z = np.linspace(0.5 + expected_avg_error / 2, 0.5 - expected_avg_error / 2, 1000)
            x = np.linspace(0, 1, 1000)
            cdf = stats.norm.cdf(z, 0.5, std_dev)
            pdf = stats.norm.pdf(x, 0.5, std_dev)
            print("P = ", cdf[0] - cdf[-1])
            print("###########################################")
            # plot normal dist
            plt.title(r'$\kappa=1.5^{' + str(i[0]) + r'}$, $\sigma=$' + str(round(std_dev, 6)))
            plt.plot(0.5 + expected_avg_error / 2, 0.0, 'r|', markersize=20)
            plt.plot(0.5 - expected_avg_error / 2, 0.0, 'r|', markersize=20)
            plt.plot(x, pdf)
            plt.annotate(r'$x_+$', xy=(0.5 + expected_avg_error / 2, 0.0),
                         textcoords='offset points', xytext=(0, 15), ha='center', size=10)
            plt.annotate(r'$x_-$', xy=(0.5 - expected_avg_error / 2, 0.0),
                         textcoords='offset points', xytext=(0, 15), ha='center', size=10)
            plt.margins(0.0)
            if np.isinf(np.max(pdf)) or np.isnan(np.max(pdf)):
                plt.ylim(top=50)
            else:
                plt.ylim(top=np.max(pdf) * 1.1)
            plt.xlabel(r'$\tilde{x}$')
            plt.show()

            rows.append([condition, d, num_nan, '', '', *q_errors])
            error.append(d)
            nan.append(num_nan)

        test_writer.writerows(rows)
        test_writer.writerows([[], [], ['Condition number', 'X_1', 'X_2']])
        test_writer.writerows(result_rows)

        # Plot all average errors
        fig, x1 = plt.subplots()
        x2 = x1.twinx()  # instantiate a second axes that shares the same x-axis

        x1.plot(x_size, error, color='black', markerfacecolor='black', linestyle=':', marker='s')
        x1.set_xlabel('Condition number')
        x1.set_ylabel('Average error')
        # x1.set_xticks(x_size)
        x1.set_xscale('log', base=condition_base)

        color = 'grey'
        x2.set_ylabel('Times the measurement is invalid', color=color)  # we already handled the x-label with ax1
        x2.bar(x_size, nan, color=color, alpha=0.150, width=10)
        x2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

        end = datetime.now()
        print("\nExecution time: ", end - start)
        print("End: ", end)


if __name__ == '__main__':
    main()
