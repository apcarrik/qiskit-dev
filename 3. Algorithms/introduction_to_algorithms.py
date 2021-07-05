# Create VQE Optimizer circuit to find minimum eigenvalue
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.circuit.library import TwoLocal

num_qubits = 2
ansatz = TwoLocal(num_qubits, 'ry', 'cz')
opt = SLSQP(maxiter=1000)
vqe = VQE(ansatz, optimizer=opt)
ansatz.draw('mpl').show()

# Run algorithm on backend
from qiskit import Aer

backend = Aer.get_backend('aer_simulator_statevector')

from qiskit.utils import QuantumInstance

backend = Aer.get_backend('aer_simulator')
quantum_instance = QuantumInstance(backend=backend, shots=800, seed_simulator=99)
# vqe = VQE(ansatz, optimizer=opt, quantum_instance=quantum_instance)
# operator = #TODO: Figure out what operator is
# min_eigen = vqe.compute_minimum_eigenvalue(operator)
# print(min_eigen)

# Complete working example - create operator
from qiskit.opflow import X, Z, I

H2_op = (-1.052373245772859 * I ^ I) + \
        (0.39793742484318045 * I ^ Z) + \
        (-0.39793742484318045 * Z ^ I) + \
        (-0.01128010425623538 * Z ^ Z) + \
        (0.18093119978423156 * X ^ X)

# run VQE and print result object it returns
from qiskit.utils import algorithm_globals

seed = 50
algorithm_globals.random_seed = seed
qi = QuantumInstance(Aer.get_backend('statevector_simulator'), seed_transpiler=seed, seed_simulator=seed)

ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
slsqp = SLSQP(maxiter=1000)
vqe = VQE(ansatz, optimizer=slsqp, quantum_instance=qi)
result = vqe.compute_minimum_eigenvalue(H2_op)
print(result)


### Using VQE Setting QuantumInstance after Construciton

# Create VQE instance without quantum instance
algorithm_globals.random_seed = seed
qi = QuantumInstance(Aer.get_backend('statevector_simulator'), seed_transpiler=seed, seed_simulator=seed)

ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
slsqp = SLSQP(maxiter=1000)
vqe = VQE(ansatz, optimizer=slsqp)

vqe.quantum_instance = qi
result = vqe.compute_minimum_eigenvalue(H2_op)
print(result)