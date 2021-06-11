import numpy as np
from qiskit import *
from matplotlib import pyplot

# Initialize quantum circuit with 3 qubits (and no classical bits)
circ = QuantumCircuit(3)

# Set circuit into 3-qubit GHZ state
circ.h(0)
circ.cx(0,1)
circ.cx(0,2)

# Visualize circuit
circ.draw('mpl')
pyplot.show()

### Simulating circuits with Qiskit Aer
from qiskit import Aer
backend = Aer.get_backend('statevector_simulator')
job = execute(circ, backend)
result = job.result()
outputstate = result.get_statevector(circ, decimals=3)
print(outputstate)

from qiskit.visualization import plot_state_city
plot_state_city(outputstate)
pyplot.show()

### Unitary Backend
backend = Aer.get_backend('unitary_simulator')
job = execute(circ, backend)
result = job.result()

print(result.get_unitary(circ, decimals=3))

### Open QASM Backend
# New circuit with classical bits
meas = QuantumCircuit(3,3)
meas.barrier(range(3))
meas.measure(range(3),range(3))
qc = circ + meas
qc.draw('mpl')
pyplot.show()

# Simulate
backend_sim = Aer.get_backend('qasm_simulator')
job_sim = execute(qc, backend_sim, shots=1024)
result_sim = job_sim.result()

counts = result_sim.get_counts(qc)
print(counts)

from qiskit.visualization import plot_histogram
plot_histogram(counts)
pyplot.show()