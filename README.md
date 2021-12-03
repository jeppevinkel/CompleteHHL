# CompleteHHL

## Description

Complete implementation of the HHL algorithm with tests, as well as tests using the Qiskit implementation of HHL.  

For validating results there's also an implementation of the classical LU-decomposition to compare results.
The validation is done by comparing the filtered measurement result to the normalized classical solution ![formula](https://render.githubusercontent.com/render/math?math=\hat{x}), 
where each element of ![\hat{x}](https://latex.codecogs.com/svg.image?\hat{x})  is squared. This is because ![img](https://bit.ly/2FMBvTL)] corresponds to the amplitudes of the
quantum state.

## Usage

To test the different implementations, we've made a Tests class in `tests.py` with matrix definitions and functions to 
run both our implementation and the implementation from Qiskit.  
Test code for running the tests is shown in the `main.py` file.