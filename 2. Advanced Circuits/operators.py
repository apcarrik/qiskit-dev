import numpy as np

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, BasicAer
from qiskit.compiler import transpile
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.quantum_info import process_fidelity

from qiskit.extensions import RXGate, XGate, CXGate

### Operator Class

# Create two-qubit Pauli-XX operator
XX = Operator([[0,0,0,1], [0,0,1,0], [0,1,0,0], [1,0,0,0]])

# Show operator properties
print(XX)
print(XX.data)
input_dim, output_dim = XX.dim
print(input_dim, output_dim)

# See input and output dimensions
op = Operator(np.random.rand(2**1, 2**2))
print('Input dimensions: ', op.input_dims())
print('Output dimensions: ', op.output_dims())

# If input matrix is not divisible into qubit systems,
# it will be stored as single-qubit operator
op = Operator(np.random.rand(6, 6))
print('Input dimensions: ', op.input_dims())
print('Output dimensions: ', op.output_dims())

# The input and output can also be manually specified when initializing new operator
op = Operator(np.random.rand(2**1, 2**2), input_dims=[4])
print('Input dimensions: ', op.input_dims())
print('Output dimensions: ', op.output_dims())

op = Operator(np.random.rand(6, 6), # system is qubit and qutrit
              input_dims=[2,3], output_dims=[2,3])
print('Input dimensions: ', op.input_dims())
print('Output dimensions: ', op.output_dims())

# extract input/output dimensions of subset of subsystem
print('Dimension of input system 0:', op.input_dims([0]))
print('Dimension of input system 1:', op.input_dims([1]))

### Converting Classes to Operators

# Create operator from a Pauli object
pauliXX = Pauli('XX')
op = Operator(pauliXX)
print(op)

# Create operator from a Gate object
op = Operator(CXGate())
print(op)

# Create operator from a parameterized Gate object
op = Operator(RXGate(np.pi/2))
print(op)

# Create operator from a QuantumCircuit object
circ = QuantumCircuit(10)
circ.h(0)
for j in range(1,10):
    circ.cx(j-1, j)
op = Operator(circ) #converts circuit into operator by implicit unitary simulation
print(op)


### Using Operators in circuits

from matplotlib import pyplot
# Create new operator
XX = Operator(Pauli('XX'))
circ = QuantumCircuit(2,2)
circ.append(XX, [0,1])
circ.measure([0,1],[0,1])
circ.draw('mpl')
pyplot.show()

# Can also directly insert Pauli object into circuit, to convert into sequence of
# single-qubit Pauli gates
backend = BasicAer.get_backend('qasm_simulator')
job = execute(circ, backend, basis_gates=['u1','u2','u3','cx'])
job.result().get_counts(0)
circ2 = QuantumCircuit(2,2)
circ2.append(Pauli('XX'), [0,1])
circ2.measure([0,1],[0,1])
circ2.draw('mpl')
pyplot.show()

### Combining Operators

# Tensor Product
A = Operator(Pauli('X'))
B = Operator(Pauli('Z'))
tens = A.tensor(B)
print(tens)

# Tensor Expansion
A = Operator(Pauli('X'))
B = Operator(Pauli('Z'))
tens = A.expand(B)
print(tens)

# Composition (matrix multiplication B.A)
A = Operator(Pauli('X'))
B = Operator(Pauli('Z'))
comp = A.compose(B)
print(comp)

# Reverse order composition (matrix multiplication A.B)
A = Operator(Pauli('X'))
B = Operator(Pauli('Z'))
comp = A.compose(B, front=True)
print(comp)

# Subsystem Composition
# compose XZ with a 3-qubit identity operator
op = Operator(np.eye(2**3))
XZ = Operator(Pauli('XZ'))
combop = op.compose(XZ, qargs=[0,2])
print(combop)

# compose YX in front of the previous operator
op = Operator(np.eye(2**3))
YX = Operator(Pauli('YX'))
combop = op.compose(XZ, qargs=[0,2], front=True)
print(combop)

# Linear Combinations
XX = Operator(Pauli('XX'))
YY = Operator(Pauli('YY'))
ZZ = Operator(Pauli('ZZ'))
op = 0.5 * (XX + YY - 3* ZZ)
print(op)
print("Operator is unitary?: ", op.is_unitary())

# Implicit conversion to Operators
op = Operator(np.eye(2)).compose([[0,1], [1,0]])
print(op)

### Comparison of Operators
print(
    Operator(Pauli('X')) == Operator(XGate())
)
# Unitaries that differ by a global phase are not considered equal
print(
    Operator(XGate()) == np.exp(1j * 0.5) * Operator(XGate())
)

# Process Fidelity - information theoretic quantification of how close two quantum
# channels are to each other, does not depend on global phase (must both be unitary)
op_a = Operator(XGate())
op_b = np.exp(1j * 0.5) * Operator(XGate())
F = process_fidelity(op_a, op_b)
print('Process fidelity = ', F)
