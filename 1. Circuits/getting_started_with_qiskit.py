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
plot_state_city(outputstate) #TODO: Figure out why this isn't working