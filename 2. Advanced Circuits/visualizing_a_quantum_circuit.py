from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from matplotlib import pyplot


### Drawing a Quantum Circuit

# Build a quantum circuit
circuit = QuantumCircuit(3,3)

circuit.x(1)
circuit.h(range(3))
circuit.cx(0,1)
circuit.measure(range(3), range(3))

# Print ASCII art version of circuit diagram to stdout
print(circuit)

### Alternative Renderers for Circuits

# Create matplotlib circuit diagram
circuit.draw('mpl')
pyplot.show()

# # Create matplotlib circuit diagram
# circuit.draw('latex')
# # TODO: this does not work - need to figure out why

### Customizing the Output

# Draw a new circuit with barriers and more registers
q_a = QuantumRegister(3, name='qa')
q_b = QuantumRegister(5, name='qb')
c_a = ClassicalRegister(3)
c_b = ClassicalRegister(5)

circuit = QuantumCircuit(q_a, q_b, c_a, c_b)

circuit.x(q_a[1])
circuit.x(q_b[1])
circuit.x(q_b[2])
circuit.x(q_b[4])
circuit.barrier()
circuit.h(q_a)
circuit.barrier(q_a)
circuit.h(q_b)
circuit.cswap(q_b[0], q_b[1], q_b[2])
circuit.cswap(q_b[2], q_b[3], q_b[4])
circuit.cswap(q_b[3], q_b[4], q_b[0])
circuit.barrier(q_b)
circuit.measure(q_a, c_a)
circuit.measure(q_b, c_b)

# Draw the circuit
circuit.draw('mpl')
pyplot.show()

# Draw the circuit with reversed bit order
circuit.draw('mpl', reverse_bits=True)
pyplot.show()

# Draw the circuit without barriers
circuit.draw('mpl', plot_barriers=False)
pyplot.show()

### Backend-specific Customizations

# Set the line length to 80 for above circuit
print(circuit)

# Change the background color in mpl
style = {'backgroundcolor': 'lightgreen'}
circuit.draw('mpl', style=style)
pyplot.show()

# Scale the mpl output to 1/2 the normal size
circuit.draw('mpl', scale=0.5)
pyplot.show()

### Display using circuit_drawer() as function

from qiskit.tools.visualization import circuit_drawer
circuit_drawer(circuit, output='mpl', plot_barriers=False)