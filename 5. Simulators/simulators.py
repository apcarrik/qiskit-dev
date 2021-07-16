import numpy as np
from pprint import pprint

# import Qiskit
from qiskit import QuantumCircuit
from qiskit import Aer, transpile
from qiskit.tools.visualization import plot_histogram, plot_state_city
import qiskit.quantum_info as qi

# show all possible backends
pprint(Aer.backends())

# use AerSimulator backend
sumulator = Aer.get_backend('aer_simulator')

## Simulating a quantum circuit

# create circuit consisting of a 2-qubit bell state |phi> = 1/2*(|00> + |11>)
circ = QuantumCircuit(2)
circ.h(0)
circ.cx(0,1)
circ.measure_all()

# transpile for simulator
simulator = Aer.get_backend('aer_simulator')
circ = transpile(circ,simulator)

# run simulation and get counts
result = simulator.run(circ).result()
counts = result.get_counts(circ)
plot_histogram(counts, title="Bell-State counts").show()

# save measurment outcomes
result = simulator.run(circ, shots=10, memory=True).result()
memory = result.get_memory(circ)
print(memory)



### Aer Simulation Options



## Simulation Method

# increase shots to reduce sampling variance
shots = 1000

# Stabilizer simulation method
sim_stabilizer = Aer.get_backend('aer_simulator_stabilizer')
job_stabilizer = sim_stabilizer.run(circ, shots=shots)
counts_stabilizer = job_stabilizer.result().get_counts(0)

# Statevector simulation method
sim_statevector = Aer.get_backend('aer_simulator_statevector')
job_statevector = sim_statevector.run(circ, shots=shots)
counts_statevector = job_statevector.result().get_counts(0)

# Density Matrix simulation method
sim_density = Aer.get_backend('aer_simulator_density_matrix')
job_density = sim_density.run(circ, shots=shots)
counts_density = job_density.result().get_counts(0)

# Matrix Product State simulation method
sim_mps = Aer.get_backend('aer_simulator_matrix_product_state')
job_mps = sim_mps.run(circ, shots=shots)
counts_mps = job_mps.result().get_counts(0)

# plot results
plot_histogram([counts_stabilizer,
                counts_statevector,
                counts_density,
                counts_mps],
               title="Counts for different simulation methods",
               legend=['stabilizer', 'statevector', 'density_matrix',
                       'matrix_product_state']).show()

## GPU Simulation

from qiskit.providers.aer import AerError

# initialize a GPU backend - note that if the target machine does not have a GPU,
# this will raise an exception.

try:
    simulator_gpu = Aer.get_backend('aer_simulator')
    simulator_gpu.set_options(device='GPU')
except AerError as e:
    print(e)

# the Aer provider will also contain preconfigured GPU simulator backends if Qiskit Aer was
# installed with GPU support on a compatible system:
# - aer_simulator_statevector_gpu
# - aer_simulator_density_matrix_gpu
# - aer_simulator_unitary_gpu

# Note: the GPU version of Aer can be installed using `pip install qiskit-aer-gpu`

## Simulation Precision

# configure a single-precision statevector simulator backend
simulator = Aer.get_backend('aer_simulator_statevector')
simulator.set_options(precision='single')

# run simulation and get counts
result = simulator.run(circ).result()
counts = result.get_counts(circ)
print(counts)




### Custom Simulator Instructions



## Saving the Simulator State

# see https://qiskit.org/documentation/tutorials/simulators/1_aer_provider.html#Saving-the-simulator-state

## Saving the Final Statevector

# construct quantum circuit wihtout measure
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0,1)
qc.save_statevector()

# transpile for simulator
simulator = Aer.get_backend('aer_simulator')
qc = transpile(qc,simulator)

# run simulation and get statevector
result = simulator.run(qc).result()
statevector = result.get_statevector(qc)
plot_state_city(statevector, title='Bell state').show()

## Saving the Circuit Unitary

# construct quantum circuit without measure
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0,1)
qc.save_unitary()

# transpile for simulator
simulator = Aer.get_backend('aer_simulator')
qc = transpile(qc, simulator)

# run simulation and get unitary
result = simulator.run(qc).result()
unitary = result.get_unitary(qc)
print("Circuit unitary:\n", unitary.round(5))

## Saving Multiple States

# construct quantum circuit without measure
steps = 5
qc = QuantumCircuit(1)
for i in range(steps):
    qc.save_statevector(label=f'psi_{i}')
    qc.rx(i * np.pi / steps, 0)
qc.save_statevector(label=f'psi_{steps}')

# transpile for simulator
simulator = Aer.get_backend('aer_simulator')
qc = transpile(qc, simulator)

# run simulation and get saved data
result = simulator.run(qc).result()
data = result.data(0)
pprint(data)



### Setting the Simulator to a Custom State



# see https://qiskit.org/documentation/tutorials/simulators/1_aer_provider.html#Setting-the-simulator-to-a-custom-state

## Setting a Custom Statevector

# generate a random statevector
num_qubits = 2
psi = qi.random_statevector(2 ** num_qubits, seed =100)

# set initial state to generated statevector
circ = QuantumCircuit(num_qubits)
circ.set_statevector(psi)
circ.save_state()

# transpile for simulator
simulator = Aer.get_backend('aer_simulator')
circ = transpile(circ, simulator)

# run simulation and get saved data
result = simulator.run(circ).result()
pprint(result.data(0))

## Use Initialize Instruction

# use intialize instruction to set initial state
circ = QuantumCircuit(num_qubits)
circ.initialize(psi, range(num_qubits))
circ.save_state()

# transpile for simulator
simulator = Aer.get_backend('aer_simulator')
circ = transpile(circ, simulator)

# run simulation and get results
result = simulator.run(circ).result()
pprint(result.data(0))

## Set Custom Density Matrix

num_qubits = 2
rho = qi.random_density_matrix(2 ** num_qubits, seed = 100)
circ = QuantumCircuit(num_qubits)
circ.set_density_matrix(rho)
circ.save_state()

# transpile for simulator
simulator = Aer.get_backend('aer_simulator')
circ = transpile(circ, simulator)

# run simulation and get saved data
result = simulator.run(circ).result()
pprint(result.data(0))

## Set Custom Unitary

# generates a random unitary
num_qubits = 2
unitary = qi.random_unitary(2** num_qubits, seed =100)

# set initial state to unitary
circ = QuantumCircuit(num_qubits)
circ.set_unitary(unitary)
circ.save_state()

# transpile for simulator
simulator = Aer.get_backend('aer_simulator')
circ = transpile(circ, simulator)

# run simulation and get saved data
result = simulator.run(circ).result()
pprint(result.data(0))