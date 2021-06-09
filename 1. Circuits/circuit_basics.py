import numpy as np
from qiskit import QuantumCircuit
from matplotlib import pyplot

circ = QuantumCircuit(3)

# Create 3-qubit GHZ state: (1/sqrt(2))*(|000> + |111>)
circ.h(0)
circ.cx(0,1)
circ.cx(1,2)

### Visualize Circuit
circ.draw('mpl')
pyplot.show()

### Simulate Circuit
from qiskit.quantum_info import Statevector

state = Statevector.from_int(0,2**3)
state = state.evolve(circ)
lat = state.draw('latex') # This is not working in Pycharm - I think this needs to be a jupyter notebook
qsp = state.draw('qsphere')
hin = state.draw('hinton')
pyplot.show()

### Unitary Representation of Circuit
from qiskit.quantum_info import Operator

U = Operator(circ)
print(U.data) # show results

### OpenQASM Backend Simulation
meas = QuantumCircuit(3,3)
meas.barrier(range(3))
meas.measure(range(3), range(3))

qc = meas.compose(circ, range(3), front=True)

qc.draw('mpl')
pyplot.show()

from qiskit import transpile
from qiskit.providers.aer import QasmSimulator

backend = QasmSimulator()
qc_compiled = transpile(qc, backend)
job_sim = backend.run(qc_compiled, shots=1024)
result_sim = job_sim.result()
counts = result_sim.get_counts(qc_compiled)
print(counts)

from qiskit.visualization import plot_histogram
plot_histogram(counts)
pyplot.show()