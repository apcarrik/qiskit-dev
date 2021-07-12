from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, assemble, Aer
from qiskit.tools.visualization import plot_histogram
from math import pi
import matplotlib.pyplot as plt



### Conditioned gates: the c_if method


# example: execute X gate if value of classical register is 0
q = QuantumRegister(1,'q')
c = ClassicalRegister(1,'c')
qc = QuantumCircuit(q,c)
qc.h(0)
qc.measure(0,0)
qc.x(0).c_if(c, 0)
qc.draw('mpl')#.show()

# example: perform bit flip on third qubit when measurement of q0 and q1 are both 1
q = QuantumRegister(3,'q')
c = ClassicalRegister(3,'c')
qc = QuantumCircuit(q,c)
qc.h(q[0])
qc.h(q[1])
qc.h(q[2])
qc.barrier()
qc.measure(q,c)

qc.x(2).c_if(c, 3) # for the 011 case
qc.x(2).c_if(c, 7) # for the 111 case

qc.draw('mpl')#.show()



### Iterative Phase Estimation (IPE)



nq = 2
m = 2
q = QuantumRegister(nq, 'q')
c = ClassicalRegister(m, 'c')

qc_S = QuantumCircuit(q,c)

## First iteration

# initialization
qc_S.h(0)
qc_S.x(1)
qc_S.draw('mpl')#.show()

# apply controlled-U gates
cu_circ = QuantumCircuit(2)
cu_circ.cp(pi/2,0,1)
cu_circ.draw('mpl')#.show()
for _ in range(2**(m-1)):
    qc_S.cp(pi/2,0,1)
qc_S.draw('mpl')#.show()


# measure in X-basis
def x_measurement(qc, qubit, cbit):
    """Measures 'qubit' in the X-basis, and stores the result in 'cbit'"""
    qc.h(qubit)
    qc.measure(qubit, cbit)

x_measurement(qc_S, q[0], c[0])
qc_S.draw('mpl')#.show()


## Subsequent iterations

# initialization with reset
qc_S.reset(0)
qc_S.h(0)
qc_S.draw('mpl')#.show()

# phase correction
qc_S.p(-pi/2,0).c_if(c,1)
qc_S.draw('mpl')#.show()

# apply control-U gates and x measurement
for _ in range(2**(m-2)):
    qc_S.cp(pi/2,0,1)

x_measurement(qc_S, q[0], c[1])
qc_S.draw('mpl')#.show()

## Execute circuit on simulator

sim = Aer.get_backend('qasm_simulator')
count0 = execute(qc_S, sim).result().get_counts()

key_new = [str(int(key,2)/2**m) for key in list(count0.keys())]
count1 = dict(zip(key_new, count0.values()))

fig, ax = plt.subplots(1,2)
plot_histogram(count0, ax=ax[0])
plot_histogram(count1, ax=ax[1])
plt.tight_layout()
plt.show()



### IPE example with a 2-qubit gate


# initialize circuit with 3 qubits
nq = 3 # number of qubits
m = 3 # number of classical bits
q = QuantumRegister(nq, 'q')
c = ClassicalRegister(m, 'c')
qc = QuantumCircuit(q,c)

## First Step

# initialization
qc.h(0)
qc.x([1,2])
qc.draw('mpl')#.show()

# apply controlled-U gates
cu_circ = QuantumCircuit(nq)
cu_circ.mcp(pi/4,[0,1], 2)
cu_circ.draw('mpl')#.show()

# apply 2^t times MCP(pi/4)
for _ in range(2**(m-1)):
    qc.mcp(pi/4,[0,1],2)
qc.draw('mpl')#.show()

# measure in x basis
x_measurement(qc, q[0], c[0])
qc.draw('mpl')#.show()

## Second Step

# initialization with reset
qc.reset(0)
qc.h(0)
qc.draw('mpl')#.show()

# phase correction
qc.p(-pi/2,0).c_if(c,1)
qc.draw('mpl')#.show()

# apply Control-U gates and measure in x basis
for _ in range(2**(m-2)):
    qc.mcp(pi/4, [0,1], 2)
x_measurement(qc, q[0], c[1])
qc.draw('mpl')#.show()

## Third Step

# initialization
qc.reset(0)
qc.h(0)

# phase correction
qc.p(-pi/4,0).c_if(c,1)
qc.p(-pi/2,0).c_if(c,2)
qc.p(-3*pi/4,0).c_if(c,3)

# controlled-U operations
for _ in range(2**(m-3)):
    qc.mcp(pi/4,[0,1],2)

# X measurement
qc.h(0)
qc.measure(0,2)
qc.draw('mpl').show()

# execute circuit on simulator
count0 = execute(qc, sim).result().get_counts()

key_new = [str(int(key,2)/2**m) for key in list(count0.keys())]
count1 = dict(zip(key_new, count0.values()))

fig, ax = plt.subplots(1,2)
plot_histogram(count0, ax=ax[0])
plot_histogram(count1, ax=ax[1])
plt.tight_layout()
plt.show()