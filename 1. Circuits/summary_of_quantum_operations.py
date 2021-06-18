import matplotlib.pyplot as plt
import numpy as np
from math import pi
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute
from qiskit.tools.visualization import circuit_drawer
from qiskit.quantum_info import state_fidelity
from qiskit import BasicAer

backend = BasicAer.get_backend('unitary_simulator')

#### === Single Qubit Gates ===
q = QuantumRegister(1)

## general unitary gate
qc = QuantumCircuit(q)
theta = pi/2
phi = pi/2
lambd = pi/2
qc.u(theta,phi,lambd,q) # previously qc.u3
qc.draw('mpl')
plt.show()
job = execute(qc, backend)
res = job.result().get_unitary(qc, decimals=3)
print("u gate:", res)

# u2 gate = u(pi/2, phi, lambda) is useful to create superpositions
qc = QuantumCircuit(q)
phi = pi/3
lambd = pi/4
qc.u(pi/2,phi,lambd,q)
qc.draw('mpl')
plt.show()
job = execute(qc, backend)
res = job.result().get_unitary(qc, decimals=3)
print("u2 gate:", res)

# u1 gate = u3(0,0,lambda) is useful to apply a quantum phase
qc = QuantumCircuit(q)
lambd = pi/5
qc.p(lambd, q)# or, alternativley qc.u(0,0,lambd,q)
qc.draw('mpl')
plt.show()
job = execute(qc, backend)
res = job.result().get_unitary(qc, decimals=3)
print("u1 gate:", res)

## Identity gate
qc = QuantumCircuit(q)
qc.id(q)
qc.draw('mpl')
plt.show()
job = execute(qc, backend)
res = job.result().get_unitary(qc, decimals=3)
print("Identity gate:", res)

## Pauli gates
# X (bit-flip) gate = u(pi,0,pi)
qc = QuantumCircuit(q)
qc.x(q)
qc.draw('mpl')
plt.show()
job = execute(qc, backend)
res = job.result().get_unitary(qc, decimals=3)
print("X gate:", res)

# Y (bit- and phase-flip) gate = u(pi,pi/2,pi/2)
qc = QuantumCircuit(q)
qc.y(q)
qc.draw('mpl')
plt.show()
job = execute(qc, backend)
res = job.result().get_unitary(qc, decimals=3)
print("Y gate:", res)

# Z (bit-flip) gate = u(pi,0,pi)
qc = QuantumCircuit(q)
qc.x(q)
qc.draw('mpl')
plt.show()
job = execute(qc, backend)
res = job.result().get_unitary(qc, decimals=3)
print("Z gate", res)

## Clifford Gates
# Hadamard gate = u3(pi/2, 0, pi)
qc = QuantumCircuit(q)
qc.h(q)
qc.draw('mpl')
plt.show()
job = execute(qc, backend)
res = job.result().get_unitary(qc, decimals=3)
print("Hadamard gate", res)

# S (sqrt(Z) phase) gate = u1(pi/2)
qc = QuantumCircuit(q)
qc.s(q)
qc.draw('mpl')
plt.show()
job = execute(qc, backend)
res = job.result().get_unitary(qc, decimals=3)
print("S gate", res)

# S-dagger (conjugate of sqrt(Z) phase) gate = u3(-pi/2)
qc = QuantumCircuit(q)
qc.sdg(q)
qc.draw('mpl')
plt.show()
job = execute(qc, backend)
res = job.result().get_unitary(qc, decimals=3)
print("S-dagger gate", res)

## C3 Gates
# T (sqrt(S)) gate = u1(pi/4)
qc = QuantumCircuit(q)
qc.t(q)
qc.draw('mpl')
plt.show()
job = execute(qc, backend)
res = job.result().get_unitary(qc, decimals=3)
print("T gate", res)

# T-dagger (conjugate of sqrt(S)) gate = u1(-pi/4)
qc = QuantumCircuit(q)
qc.tdg(q)
qc.draw('mpl')
plt.show()
job = execute(qc, backend)
res = job.result().get_unitary(qc, decimals=3)
print("T-dagger gate", res)

## Standard Rotations
# Rotation around X-axis = u3(theta, -pi/2, pi/2)
qc = QuantumCircuit(q)
theta=pi/2
qc.rx(theta, q)
qc.draw('mpl')
plt.show()
job = execute(qc, backend)
res = job.result().get_unitary(qc, decimals=3)
print("X-axis rotation gate", res)

# Rotation around Y-axis = u3(theta, 0, 0)
qc = QuantumCircuit(q)
theta=pi/2
qc.ry(theta, q)
qc.draw('mpl')
plt.show()
job = execute(qc, backend)
res = job.result().get_unitary(qc, decimals=3)
print("Y-axis rotation gate", res)

# Rotation around Z-axis = u1(phi)
qc = QuantumCircuit(q)
phi=pi/2
qc.rz(phi, q)
qc.draw('mpl')
plt.show()
job = execute(qc, backend)
res = job.result().get_unitary(qc, decimals=3)
print("Z-axis rotation gate", res)



#### === Multi-Qubit Gates ===

### Two-Qubit Gates
q= QuantumRegister(2)

## Controlled Pauli Gates
# Controled-X Gate
qc = QuantumCircuit(q)
qc.cx(q[0], q[1])
qc.draw('mpl')
plt.show()
job = execute(qc, backend)
res = job.result().get_unitary(qc, decimals=3)
print("Controlled-X gate", res)

# Controled-Y Gate
qc = QuantumCircuit(q)
qc.cy(q[0], q[1])
qc.draw('mpl')
plt.show()
job = execute(qc, backend)
res = job.result().get_unitary(qc, decimals=3)
print("Controlled-Y gate", res)

# Controled-Z Gate
qc = QuantumCircuit(q)
qc.cz(q[0], q[1])
qc.draw('mpl')
plt.show()
job = execute(qc, backend)
res = job.result().get_unitary(qc, decimals=3)
print("Controlled-Z gate", res)

## Controled Hadamard Gate
qc = QuantumCircuit(q)
qc.ch(q[0], q[1])
qc.draw('mpl')
plt.show()
job = execute(qc, backend)
res = job.result().get_unitary(qc, decimals=3)
print("Controlled Hadamard gate", res)

## Controlled Rotation Gates
# Controled-Z Rotation Gate
qc = QuantumCircuit(q)
lambd = pi/2
qc.crz(lambd, q[0], q[1])
qc.draw('mpl')
plt.show()
job = execute(qc, backend)
res = job.result().get_unitary(qc, decimals=3)
print("Controlled Rotation Around Z-axis gate", res)

# Controled-Phase Rotation Gate
qc = QuantumCircuit(q)
lambd = pi/2
qc.cp(lambd, q[0], q[1]) # previously qc.cu1
qc.draw('mpl')
plt.show()
job = execute(qc, backend)
res = job.result().get_unitary(qc, decimals=3)
print("Controlled-Phase Roatation gate", res)

# Controled-u Rotation Gate
qc = QuantumCircuit(q)
theta = pi/2
phi = pi/2
lambd = pi/2
qc.cu(theta,phi,lambd, 0, q[0], q[1]) # previously qc.cu3
qc.draw('mpl')
plt.show()
job = execute(qc, backend)
res = job.result().get_unitary(qc, decimals=3)
print("Controlled-u Rotation gate", res)

# SWAP Gate
qc = QuantumCircuit(q)
qc.swap(q[0], q[1])
qc.draw('mpl')
plt.show()
job = execute(qc, backend)
res = job.result().get_unitary(qc, decimals=3)
print("SWAP gate", res)

### Three-Qubit Gates
q= QuantumRegister(3)

## Toffoli Gate
qc = QuantumCircuit(q)
qc.ccx(q[0], q[1], q[2])
qc.draw('mpl')
plt.show()
job = execute(qc, backend)
res = job.result().get_unitary(qc, decimals=3)
print("Toffoli gate", res)

## Controlled Swap (Fredkin) Gate
qc = QuantumCircuit(q)
qc.cswap(q[0], q[1], q[2])
qc.draw('mpl')
plt.show()
job = execute(qc, backend)
res = job.result().get_unitary(qc, decimals=3)
print("Controlled Swap gate", res)



#### === Non-Unitary Operations ===
q = QuantumRegister(1)
c = ClassicalRegister(1)

### Measurements
# qubit in state |0> only
qc = QuantumCircuit(q,c)
qc.measure(q,c)
qc.draw('mpl')
plt.show()
backend = BasicAer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=1024)
res = job.result().get_counts(qc)
print("Measurement Results: ", res)

# qubit in state |0> and |1> with equal probability
qc = QuantumCircuit(q,c)
qc.h(q)
qc.measure(q,c)
qc.draw('mpl')
plt.show()
backend = BasicAer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=1024)
res = job.result().get_counts(qc)
print("Measurement Results: ", res)

### Reset
# Resets all qubits to |0> state
qc = QuantumCircuit(q,c)
qc.h(q)
qc.reset(q[0])
qc.measure(q,c)
qc.draw('mpl')
plt.show()
backend = BasicAer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=1024)
res = job.result().get_counts(qc)
print("Measurement Results: ", res)

### Conditional Operators
# classical bit always takes value 0 so qubit state is always flipped
qc = QuantumCircuit(q,c)
qc.x(q[0]).c_if(c,0)
qc.measure(q,c)
qc.draw('mpl')
plt.show()
backend = BasicAer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=1024)
res = job.result().get_counts(qc)
print("Measurement Results: ", res)

# classical bit by first measurement is random, but conditional operation results in qubit being
#   deterministically put to 1.
qc = QuantumCircuit(q,c)
qc.h(q)
qc.measure(q,c)
qc.x(q[0]).c_if(c,0)
qc.measure(q,c)
qc.draw('mpl')
plt.show()
backend = BasicAer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=1024)
res = job.result().get_counts(qc)
print("Measurement Results: ", res)

### Arbitrary Initialization
# Initialize qubit register to arbitrary state:
# |psi> = i/4|000> + 1/sqrt(8)|001> + (1+i)/4|010> + (1 + 2i)/sqrt(8)|101> + 1/4|110>
import math
desired_vector = [
    1 / math.sqrt(16) * complex(0, 1),
    1 / math.sqrt(8) * complex(1, 0),
    1 / math.sqrt(16) * complex(1, 1),
    0,
    0,
    1 / math.sqrt(8) * complex(1, 2),
    1 / math.sqrt(16) * complex(1, 0),
    0
]

q = QuantumRegister(3)
qc = QuantumCircuit(q)
qc.initialize(desired_vector, [q[0],q[1],q[2]])
qc.draw('mpl')
plt.show()
backend = BasicAer.get_backend('statevector_simulator')
job = execute(qc, backend)
qc_state = job.result().get_statevector(qc)
print("QC state:", qc_state)

# you should check the Fidelity to ensure the state matches the desired vector
fid = state_fidelity(desired_vector, qc_state)
print("Fidelity: ", fid)
