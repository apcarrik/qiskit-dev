from qiskit import IBMQ, transpile
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit.tools.visualization import plot_histogram

## Terra Mock Backends
from qiskit.test.mock import FakeVigo
device_backend = FakeVigo()

# Build test circuit
circ = QuantumCircuit(3,3)
circ.h(0)
circ.cx(0,1)
circ.cx(1,2)
circ.measure([0,1,2], [0,1,2])

sim_ideal = AerSimulator()

# Execute circuit and get counts
result = sim_ideal.run(transpile(circ,sim_ideal)).result()
counts = result.get_counts(0)
plot_histogram(counts, title="Ideal counts for 3-qubit GHZ state").show()

## Generate Simulator that Mimics Real Device

sim_vigo = AerSimulator.from_backend(device_backend)

# Run noisy simulation
tcirc = transpile(circ, sim_vigo)
result_noise = sim_vigo.run(tcirc).result()
counts_noise = result_noise.get_counts()
plot_histogram(counts_noise, title="Counts for 3-qubit GHZ state with noisy model").show()