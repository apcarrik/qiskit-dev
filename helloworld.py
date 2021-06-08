'''
helloworld.py

This file is intended as a simple template for qiskit files.
The code can be found at: https://qiskit.org/documentation/intro_tutorial1.html
'''

### 1. Import Packages

import numpy as np
from matplotlib import pyplot
from qiskit import QuantumCircuit, transpile # instructions of quantum system
from qiskit.providers.aer import QasmSimulator # Aer high performance circuit simulator
from qiskit.visualization import plot_histogram # creates histograms of circuit output

### 2. Initialize Variables

circuit = QuantumCircuit(2,2) # initialize the 2 qubits in zero state, and 2 classical bits set to 0

### 3. Add gates

circuit.h(0) # add Hadamard gate on qubit 0
circuit.cx(0, 1) # add CNOT gate with control qubit 0 and target qubit 1
circuit.measure([0,1], [0,1]) # Measure the two qubits and saves the result to the two classical bits
# Note: this circuit implements a simple Bell state, where there is equal chance of measuring |00> and |11>

### 4. Visualize the Circuit

circuit.draw(output='mpl') # draw the circuit as matplotlib figure
pyplot.show() # show the figure

### 5. Simulate the Experiment

simulator = QasmSimulator() # Create instance of Aer's qasm_simulator
compiled_circuit = transpile(circuit, simulator) # Compile the circuit down to low-level QASM instructions.
                                                 # Supported by the backend (not needed for simple circuits)
job = simulator.run(compiled_circuit, shots=1000) # Execute circuit on qasm simulator
result = job.result() # Fetch results
counts = result.get_counts(circuit) # Get the counts of outcomes from results
print("\nTotal count for 00 and 11 are:", counts) # Print total count for each outcome

### 6. Visualize the Results
plot_histogram(counts) # plots the results as a histogram
pyplot.show() # show the figure