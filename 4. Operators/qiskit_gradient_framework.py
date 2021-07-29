import numpy as np

from qiskit.opflow import Z, X, I, StateFn, CircuitStateFn, SummedOp
from qiskit.opflow.gradients import Gradient, NaturalGradient, QFI, Hessian

from qiskit.circuit import (QuantumCircuit, QuantumRegister, Parameter,
                            ParameterVector, ParameterExpression)
from qiskit.circuit.library import EfficientSU2



### First Order Gradients


## Gradients with respect to Measurement Operator Parameters

# instantiate the quantum state
a = Parameter('a')
b = Parameter('b')
q = QuantumRegister(1)
qc = QuantumCircuit(q)
qc.h(q)
qc.rz(a, q[0])
qc.rx(b, q[0])

# instantiate Hamiltonian observable
H = (2 * X) + Z

# combine the Hamiltonian observable and the state
op = ~StateFn(H) @ CircuitStateFn(primitive=qc, coeff=1.)

# print the operator corresponding to the expectation value
print(op)

params = [a, b]

# define the values to be asigned to the parameters
value_dict = {a: np.pi / 4, b: np.pi}

# convert the operator and the gradient target params into the respective operators
grad = Gradient().convert(operator= op, params= params)

# print the operator corresponding to the gradient
print(grad)

# assign the parameters and evaluate the gradient
grad_result = grad.assign_parameters(value_dict).eval()
print("Gradient", grad_result)


## Gradients with respect to State Parameters

# define the Hamiltonian with fixed coefficients
H = 0.5 * X - 1 * Z

# define the parameters w.r.t we want to compute the gradients
params = [a,b]

# define the values to be assigned to the parameters
value_dict = {a: np.pi / 4, b: np.pi}

# combine the Hamiltonian observable and the state
op = ~StateFn(H) @ CircuitStateFn(primitive=qc, coeff=1.)

# print the operator corresponding to the expectation value
print(op)

## Parameter Shift Gradients

# convert the expectation value into an operator corrsponding to the gradient w.r.t
# the state parameters using the parameter shift method
state_grad = Gradient(grad_method='param_shift').convert(operator=op, params=params)

# print the operator corresponding to the gradient
print(state_grad)

# assign the parameters and evaluate the gradient
state_grad_result = state_grad.assign_parameters(value_dict).eval()
print('State gradient computed with parameter shift', state_grad_result)

## Linear Combination of Unitaries Gradients

# convert the expectation value into an operator corresponding to the gradient w.r.t
# the state parameter using a linear combination of unitaries method
state_grad = Gradient(grad_method='lin_comb').convert(operator=op, params=params)

# print the operator corresponding to the gradient
print(state_grad)

# assign the parameters and evaluate the gradient
state_grad_result = state_grad.assign_parameters(value_dict).eval()
print('State gradient computed with the linear combination method', state_grad_result)

## Finite Difference Gradients

# while the previous methods were analytical, this is a numerical approximation method

# convert the expectation value into an operator corresponding to the gradient w.r.t
# the state parameter using the finite difference method
state_grad = Gradient(grad_method='lin_comb').convert(operator=op, params=params)

# print the operator corresponding to the gradient
print(state_grad)

# assign the parameters and evaluate the gradient
state_grad_result = state_grad.assign_parameters(value_dict).eval()
print('State gradient computed with the linear combination method', state_grad_result)


## Natural Gradients

# besides the method to compute the circuit gradients resp. QFI, a regularization method
# can be chosen: 'ridge' or 'lasso' with automatic parameter search or 'perturb_diag'
# or 'perturb_diag_elements', which perturb the diagonal elements of the QFI
nat_grad = NaturalGradient(grad_method='lin_comb', qfi_method='lin_comb_full',
                           regularization='ridge').convert(operator=op, params=params)

# assign the parameters and evaluate the gradient
nat_grad_result = nat_grad.assign_parameters(value_dict).eval()
print('Natural gradient computed with the linear combination of unitaries', nat_grad_result)



### Hessians (Second Order Gradients)



## Hessians w.r.t Measurement Operator Parameters

# instantiate the Hamiltonian observable
H = X

# instantiate the quantum state with two parameters
a = Parameter('a')
b = Parameter('b')

q = QuantumRegister(1)
qc = QuantumCircuit(q)
qc.h(q)
qc.rz(a, q[0])
qc.rx(b, q[0])

# combine the Hamiltonian observable and the state
op = ~StateFn(H) @ CircuitStateFn(primitive=qc, coeff=1.)

# convert the operator and the Hessian target coefficients into the respective operator
hessian = Hessian().convert(operator=op, params=[a,b])

# define the values to be assigned to the parameters
value_dict = {a: np.pi/4, b: np.pi/4}

# assign the parameters and evaluate the Hessian w.r.t the Hamiltonian coefficients
hessian_result = hessian.assign_parameters(value_dict).eval()
print('Hessian \n', np.real(np.array(hessian_result)))

## Hessians w.r.t. State Parameters

# define parameters
params = [a, b]

# get the operator object representing the Hessian
state_hess = Hessian(hess_method='param_shift').convert(operator=op, params=params)

# assign the parameters and evaluate the Hessian
hessian_result = state_hess.assign_parameters(value_dict).eval()
print("Hessian computed using the parameter shift method\n", (np.array(hessian_result)))

# get the operator object representing the Hessian
state_hess = Hessian(hess_method='lin_comb').convert(operator=op, params=params)

# assign the parameters and evaluate the Hessian
hessian_result = state_hess.assign_parameters(value_dict).eval()
print("Hessian computed using the linear combination of unitaries method\n",
      (np.array(hessian_result)))

# get the operator object representing the Hessian using finite difference
state_hess = Hessian(hess_method='fin_diff').convert(operator=op, params=params)

# assign the parameters and evaluate the Hessian
hessian_result = state_hess.assign_parameters(value_dict).eval()
print("Hessian computed using the finite difference method\n",
      (np.array(hessian_result)))



### Quantum Fisher Information (QFI)



## Linear Combination Full QFI

# wrap the quantum circuit into a CircuitStateFn
state = CircuitStateFn(primitive=qc, coeff=1.)

# convert the state and the parameters into the operator object that represents the QFI
qfi = QFI(qfi_method='lin_comb_full').convert(operator=state, params=params)

# define the values for which the QFI is to be computed
values_dict = {a: np.pi/4, b: 0.1}

# assign the parameters and evaluate the QFI
qfi_result = qfi.assign_parameters(values_dict).eval()
print('full QFI \n', np.real(np.array(qfi_result)))

## Block-diagonal and Diagonal Approximation

# convert the state and the parameters into the operator object that represents the QFI
# and set the approximation to 'block_diagonal'
qfi = QFI(qfi_method='overlap_block_diag').convert(operator=state, params=params)

# assign the parameters and evaluate the QFI
qfi_result = qfi.assign_parameters(values_dict).eval()
print('Block-diagonal QFI \n', np.real(np.array(qfi_result)))

# convert the state and the parameters into the operator object that represents the QFI
# and set the approximation to 'diagonal'
qfi = QFI(qfi_method='overlap_diag').convert(operator=state, params=params)

# assign the parameters and evaluate the QFI
qfi_result = qfi.assign_parameters(values_dict).eval()
print('Diagonal QFI \n', np.real(np.array(qfi_result)))


### Application Example: VQE with Gradient-Based Optimization


# execution imports
from qiskit import Aer
from qiskit.utils import QuantumInstance

# algorithm imports
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import CG

from qiskit.opflow import I, X , Z
from qiskit.circuit import QuantumCircuit, ParameterVector
from scipy.optimize import minimize

# initiate the system Hamiltonian
h2_hamiltonian = -1.05 * (I ^ I) + 0.39 * (I ^ Z) - 0.39 * (Z ^ I) - 0.01 * (Z ^ Z) + \
    0.18 * (X ^ X)

# this is the target energy
h2_energy = -1.85727503

# define the Ansatz
wavefunction = QuantumCircuit(2)
params = ParameterVector('theta', length=8)
it = iter(params)
wavefunction.ry(next(it), 0)
wavefunction.ry(next(it), 1)
wavefunction.rz(next(it), 0)
wavefunction.rz(next(it), 1)
wavefunction.cx(0, 1)
wavefunction.ry(next(it), 0)
wavefunction.ry(next(it), 1)
wavefunction.rz(next(it), 0)
wavefunction.rz(next(it), 1)

# define the expectation value corresponding to the energy
op = ~StateFn(h2_hamiltonian) @ StateFn(wavefunction)

# choose wheither VQE should use Gradient or NaturalGradient
grad = Gradient(grad_method='lin_comb')

qi_sv = QuantumInstance(Aer.get_backend('aer_simulator_statevector'),
                        shots=1,
                        seed_simulator=2,
                        seed_transpiler=2)

# configure Gradient algorithm
optimizer = CG(maxiter=50)

# Gradient callable
vqe = VQE(wavefunction, optimizer=optimizer, gradient=grad, quantum_instance=qi_sv)

result = vqe.compute_minimum_eigenvalue(h2_hamiltonian)
print("Result:", result.optimal_value, 'Reference:', h2_energy)

# define QuantumInstance to execute quantum circuits and run algorithm


