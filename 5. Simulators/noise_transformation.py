from qiskit.providers.aer.utils import approximate_quantum_error
from qiskit.providers.aer.utils import approximate_noise_model

import numpy as np

# Import Aer QuantumError functions that will be used
from qiskit.providers.aer.noise import amplitude_damping_error
from qiskit.providers.aer.noise import reset_error
from qiskit.providers.aer.noise import pauli_error



### Approximating Amplitude Damping Noise With Reset Noise




gamma = 0.23
error = amplitude_damping_error(gamma)
results = approximate_quantum_error(error, operator_string="reset") # this function is throwing an error

print(results)
p = (1 + gamma - np.sqrt(1 - gamma)) / 2
q = 0

print("")
print("Expected results:")
print("P(0) = {}".format(1-(p+q)))
print("P(1) = {}".format(p))
print("P(2) = {}".format(q))




### Different Input Types




gamma = 0.23
K0 = np.array([[1,0],[0,np.sqrt(1-gamma)]])
K1 = np.array([[0,np.sqrt(gamma)],[0,0]])
results = approximate_quantum_error((K0,K1), operator_string="reset") # still not working
print(results)

reset_to_0 = [np.array([[1,0],[0,0]]), np.array([[0,1],[0,0]])]
reset_to_1 = [np.array([[0,0],[1,0]]), np.array([[0,0],[0,1]])]
reset_kraus = (reset_to_0, reset_to_1)

gamma = 0.23
error = amplitude_damping_error(gamma)
result = approximate_quantum_error(error, operator_list=reset_kraus)
print(results)
