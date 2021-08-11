### Using the Matrix Product State Simulation Method



import numpy as np

# import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import AerSimulator

# construct quantum circuit
circ = QuantumCircuit(2,2)
circ.h(0)
circ.cx(0,1)
circ.measure([0,1],[0,1])

# select the AerSimulator from the Aer provider
simulator = AerSimulator(method='matrix_product_state')

# run and get counts
tcirc = transpile(circ, simulator)
result = simulator.run(tcirc).result()
counts = result.get_counts(0)
print(counts)

# construct quantum circuit
circ = QuantumCircuit(2,2)
circ.h(0)
circ.cx(0,1)

# define a snapshot that shows the current state vector
circ.save_statevector(label='my_sv')
circ.save_matrix_product_state(label='my_mps')
circ.measure([0,1],[0,1])

# execute and get saved data
tcirc = transpile(circ, simulator)
result = simulator.run(tcirc).result()
data = result.data(0)
print(data)



### Testing Large EPR State Circuit



num_qubits_arr = [50, 100, 500]
for num_qubits in num_qubits_arr:
    circ = QuantumCircuit(num_qubits, num_qubits)

    # create EPR state
    circ.h(0)
    for i in range(0,num_qubits-1):
        circ.cx(i,i+1)

    # measure
    circ.measure(range(num_qubits), range(num_qubits))
    tcirc = transpile(circ, simulator)
    result = simulator.run(tcirc).result()
    print("\nFor {} qubits:".format(num_qubits))
    print("\tTime taken: {}s".format(result.time_taken))
    print("\tCounts: ", counts)
