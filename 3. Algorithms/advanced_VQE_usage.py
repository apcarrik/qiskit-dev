from qiskit import Aer
from qiskit.opflow import X, Z, I
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.circuit.library import TwoLocal

H2_op = (-1.052373245772859 * I ^ I) + \
        (0.39793742484318045 * I ^ Z) + \
        (-0.39793742484318045 * Z ^ I) + \
        (-0.01128010425623538 * Z ^ Z) + \
        (0.18093119978423156 * X ^ X)

# Run optimization and use optimal point as initial point to restart optimization from
seed = 50
algorithm_globals.random_seed = seed
qi = QuantumInstance(Aer.get_backend('statevector_simulator'),
                     seed_transpiler=seed,
                     seed_simulator=seed)

ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
slsqp = SLSQP(maxiter=1000)
vqe = VQE(ansatz,
          optimizer=slsqp,
          quantum_instance=qi)
result = vqe.compute_minimum_eigenvalue(H2_op)
print(result)
optimizer_evals = result.optimizer_evals

initial_point = result.optimal_point
seed = 50
algorithm_globals.random_seed = seed
qi = QuantumInstance(Aer.get_backend('statevector_simulator'),
                     seed_transpiler=seed,
                     seed_simulator=seed)

ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
slsqp = SLSQP(maxiter=1000)
vqe = VQE(ansatz,
          optimizer=slsqp,
          initial_point=initial_point,
          quantum_instance=qi)
result1 = vqe.compute_minimum_eigenvalue(H2_op)
print(result1)
optimizer_evals1 = result1.optimizer_evals
print(f'Optimizerevals is {optimizer_evals1} with initial point versus {optimizer_evals} without initial point.')


### Expectation

# Create an expectation object with sample noise to mimic real computations

algorithm_globals.random_seed = seed
qi = QuantumInstance(Aer.get_backend('aer_simulator'),
                     seed_transpiler=seed,
                     seed_simulator=seed)
ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
slsqp = SLSQP(maxiter=1000)
vqe = VQE(ansatz,
          optimizer=slsqp,
          quantum_instance=qi,
          include_custom=True)
result = vqe.compute_minimum_eigenvalue(H2_op)
print(result)

# Simulation fails if include_custom is not set (exits prematurely due to noise)

algorithm_globals.random_seed = seed
qi = QuantumInstance(Aer.get_backend('aer_simulator'),
                     seed_transpiler=seed,
                     seed_simulator=seed)
ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
slsqp = SLSQP(maxiter=1000)
vqe = VQE(ansatz,
          optimizer=slsqp,
          quantum_instance=qi)
result = vqe.compute_minimum_eigenvalue(H2_op)
print(result)

# Change optimizer to SPSA, which is designed for noisy environments
from qiskit.algorithms.optimizers import SPSA

algorithm_globals.random_seed = seed
qi = QuantumInstance(Aer.get_backend('aer_simulator'),
                     seed_transpiler=seed,
                     seed_simulator=seed)
ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
spsa = SPSA(maxiter=1000)
vqe = VQE(ansatz,
          optimizer=spsa,
          quantum_instance=qi)
result = vqe.compute_minimum_eigenvalue(H2_op)
print(result)

# Create expectation and pass to VQE
from qiskit.opflow import AerPauliExpectation

algorithm_globals.random_seed = seed
qi = QuantumInstance(Aer.get_backend('aer_simulator'),
                     seed_transpiler=seed,
                     seed_simulator=seed)
ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
slsqp = SLSQP(maxiter=1000)
vqe = VQE(ansatz,
          optimizer=slsqp,
          quantum_instance=qi,
          expectation=AerPauliExpectation())
result = vqe.compute_minimum_eigenvalue(H2_op)
print(result)

# Turn Pauli grouping off on SPSA run
from qiskit.opflow import PauliExpectation

algorithm_globals.random_seed = seed
qi = QuantumInstance(Aer.get_backend('aer_simulator'),
                     seed_transpiler=seed,
                     seed_simulator=seed)
ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
spsa = SPSA(maxiter=1000)
vqe = VQE(ansatz,
          optimizer=spsa,
          quantum_instance=qi,
          expectation=PauliExpectation(group_paulis=False))
result = vqe.compute_minimum_eigenvalue(H2_op)
print(result)



### Gradient


from qiskit.providers.aer import QasmSimulator

algorithm_globals.random_seed = seed
qi = QuantumInstance(QasmSimulator(method='matrix_product_state'),
                     shots=1)
ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
slsqp = SLSQP(maxiter=1000)
vqe = VQE(ansatz,
          optimizer=slsqp,
          quantum_instance=qi,
          include_custom=True)
result = vqe.compute_minimum_eigenvalue(H2_op)
print(result)
