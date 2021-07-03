from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager

# Create simple circuit with toffoli gate
circ = QuantumCircuit(3)
circ.ccx(0,1,2)
circ.draw('mpl').show()

# Unroll circuit (show base gates)
from qiskit.transpiler.passes import Unroller
pass_ = Unroller(['u1', 'u2', 'u3', 'cx'])
pm = PassManager(pass_)
new_circ = pm.run(circ)
new_circ.draw('mpl').show()

# Show transpiler pass options
from qiskit.transpiler import passes
qis_passes = [pass_ for pass_ in dir(passes) if pass_[0].isupper()]
print(qis_passes)

### Different Variants of the Same Pass

from qiskit.transpiler import CouplingMap, Layout
from qiskit.transpiler.passes import BasicSwap, LookaheadSwap, StochasticSwap

coupling = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]

circuit = QuantumCircuit(7)
circuit.h(3)
circuit.cx(0, 6)
circuit.cx(6, 0)
circuit.cx(0, 1)
circuit.cx(3, 1)
circuit.cx(3, 0)
circuit.draw('mpl').show()

coupling_map = CouplingMap(couplinglist=coupling)

bs = BasicSwap(coupling_map=coupling_map)
pass_manager = PassManager(bs)
basic_circ = pass_manager.run(circuit)
basic_circ.draw('mpl').show()

ls = LookaheadSwap(coupling_map=coupling_map)
pass_manager = PassManager(ls)
lookahead_circ = pass_manager.run(circuit)
lookahead_circ.draw('mpl').show()

ss = StochasticSwap(coupling_map=coupling_map)
pass_manager = PassManager(ss)
stochastic_circ = pass_manager.run(circuit)
stochastic_circ.draw('mpl').show()

### Preset Pass Managers

# Optimizes circuit baed on 4 optimization levels (0-3)
import math
from qiskit.test.mock import FakeTokyo

backend = FakeTokyo()

qc = QuantumCircuit(10)

random_state = [
    1 / math.sqrt(4) * complex(0,1),
    1 / math.sqrt(8) * complex(0,1),
    0,
    0,
    0,
    0,
    0,
    0,
    1 / math.sqrt(8) * complex(1,0),
    1 / math.sqrt(8) * complex(0,1),
    0,
    0,
    0,
    0,
    1 / math.sqrt(4) * complex(1,0),
    1 / math.sqrt(8) * complex(1,0)
]

qc.initialize(random_state, range(4))
qc.draw('mpl').show()

# Map to 20 qubit Tokyo device with different optimization levels
optimized_0 = transpile(qc, backend=backend, seed_transpiler=11, optimization_level=0)
print('Optimization level: 0')
print('gates = ', optimized_0.count_ops())
print('depth = ', optimized_0.depth())

optimized_1 = transpile(qc, backend=backend, seed_transpiler=11, optimization_level=1)
print('Optimization level: 1')
print('gates = ', optimized_1.count_ops())
print('depth = ', optimized_1.depth())

optimized_2 = transpile(qc, backend=backend, seed_transpiler=11, optimization_level=2)
print('Optimization level: 2')
print('gates = ', optimized_2.count_ops())
print('depth = ', optimized_2.depth())

optimized_3 = transpile(qc, backend=backend, seed_transpiler=11, optimization_level=3)
print('Optimization level: 3')
print('gates = ', optimized_3.count_ops())
print('depth = ', optimized_3.depth())



##### Introducing the DAG #####



from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.dagcircuit import DAGCircuit

q = QuantumRegister(3, 'q')
c = ClassicalRegister(3, 'c')
circ = QuantumCircuit(q, c)
circ.h(q[0])
circ.cx(q[0], q[1])
circ.measure(q[0], c[0])
circ.rz(0.5, q[1]).c_if(c,2)
circ.draw('mpl').show()

# Show the DAG representation of circuit
from qiskit.converters import circuit_to_dag
from qiskit.tools.visualization import dag_drawer
dag = circuit_to_dag(circ)
dag_drawer(dag)

# Get all op nodes in DAG & examine Node details
op_nodes = dag.op_nodes()
print(op_nodes)
for i,node in  enumerate(dag.op_nodes()):
    print('node ', i)
    print('node name: ', node.name)
    print('node op: ', node.op)
    print('node qargs: ', node.qargs)
    print('node cargs: ', node.cargs)
    print('node condition: ', node.condition)

# Add an operation to the back
from qiskit.circuit.library import HGate
dag.apply_operation_back(HGate(), qargs=[q[0]])
dag_drawer(dag)

# Add an operation to the front
from qiskit.circuit.library import CCXGate
dag.apply_operation_front(CCXGate(), qargs=[q[0], q[1], q[2]], cargs=[])
dag_drawer(dag)

# Substitute a node within a circuit
from qiskit.circuit.library import CHGate, U2Gate, CXGate
mini_dag = DAGCircuit()
p = QuantumRegister(2, "p")
mini_dag.add_qreg(p)
mini_dag.apply_operation_back(CHGate(), qargs=[p[0], p[1]])
mini_dag.apply_operation_back(U2Gate(0.1,0.2), qargs=[ p[1]])

# Substitute the cx node with above mini-dag
cx_node = dag.op_nodes(op=CXGate).pop()
dag.substitute_node_with_dag(node=cx_node, input_dag=mini_dag, wires=[p[0], p[1]])
dag_drawer(dag)

# Convert back to regular QuantumCircuit object
from qiskit.converters import dag_to_circuit
circuit = dag_to_circuit(dag)
circuit.draw('mpl').show()


### Implementing a BasicMapper Pass

from copy import copy
from qiskit.transpiler.basepasses import  TransformationPass
from qiskit.transpiler import Layout
from qiskit.circuit.library import SwapGate

class BasicSwap(TransformationPass):
    """ Maps (with minimum effort) a DAGCircuit onto a `coupling_map` adding swap gates."""

    def __init__(self,
                 coupling_map,
                 initial_layout=None):
        """Maps a DAGCircuit onto a `coupling_map` using swap gates.

        Args:
            coupling_map (CouplingMap): Directed graph representing a coupling map.
            initial_layout (Layout): initial layout of qubits in mapping.
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.initial_layout = initial_layout

    def run(self,
            dag):
        """Runs the BasicSwap pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to map.

        Returns:
            DAGCircuit: A mapped DAG.

        Raises:
            TranspilerError: if the coupling map or the layout are not
            compatible with the DAG.
        """
        new_dag = DAGCircuit()
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        if self.initial_layout is None:
            if self.property_set["layout"]:
                self.initial_layout = self.property_set["layout"]
            else:
                self.initial_layout = Layout.generate_trivial_layout(*dag.qregs.values())

        if len(dag.qubits) != len(self.initial_layout):
            raise TranspilerError("The layout does not match the amount of qubits in the DAG")

        if len(self.coupling_map.physical_qubits) != len(self.initial_layout):
            raise TranspilerError("Mappers require to have the layout to be the same size as the coupling map")

        canonical_register = dag.qregs['q']
        trivial_layout = Layout.generate_trivial_layout(canonical_register)
        current_layout = trivial_layout.copy()

        for layer in dag.serial_layers():
            subdag = layer['graph']

            for gate in subdag.two_qubit_ops():
                physical_q0 = current_layout[gate.qargs[0]]
                physical_q1 = current_layout[gate.qargs[1]]
                if self.coupling_map.distance(physical_q0, physical_q1) != 1:
                    # Insert a new layer with the SWAP(s).
                    swap_layer = DAGCircuit()
                    swap_layer.add_qreg(canonical_register)

                    path = self.coupling_map.shortest_undirected_path(physical_q0,physical_q1)
                    for swap in range(len(path) - 2):
                        connected_wire_1 = path[swap]
                        connected_wire_2 = path[swap+1]

                        qubit_1 = current_layout[connected_wire_1]
                        qubit_2 = current_layout[connected_wire_2]

                        # create the swap operation
                        swap_layer.apply_operation_back(SwapGate(),
                                                        qargs=[qubit_1, qubit_2],
                                                        cargs=[])

                    # layer insertion
                    order = current_layout.reorder_bits(new_dag.qubits)
                    new_dag.compose(swap_layer, qubits=order)

                    # update current_layout
                    for swap in range(len(path) - 2):
                        current_layout.swap(path[swap], path[swap+1])
            order = current_layout.reorder_bits(new_dag.qubits)
            new_dag.compose(subdag, qubits=order)
        return new_dag


### Test pass on example circuit
q = QuantumRegister(7, 'q')
in_circ = QuantumCircuit(q)
in_circ.h(q[0])
in_circ.cx(q[0], q[4])
in_circ.cx(q[2], q[3])
in_circ.cx(q[6], q[1])
in_circ.cx(q[5], q[0])
in_circ.rz(0.1, q[2])
in_circ.cx(q[5], q[0])

# Create pass manager, pass example circuit to it, and obtain transformed circuit
from qiskit.transpiler import PassManager
from qiskit.transpiler import CouplingMap
from qiskit import BasicAer
pm = PassManager()
coupling = [[0,1], [1,2], [2,3], [3,4], [4,5], [5,6]]
coupling_map = CouplingMap(couplinglist=coupling)

pm.append([BasicSwap(coupling_map)])
out_circ = pm.run(in_circ)
in_circ.draw('mpl').show()
out_circ.draw('mpl').show()


### Transpiler Logging

# Set up Python logging
import logging

logging.basicConfig(level='DEBUG')

# Make test circuit to see debug log statements
from qiskit.test.mock import FakeTenerife

log_circ = QuantumCircuit(2,2)
log_circ.h(0)
log_circ.h(1)
log_circ.h(1)
log_circ.x(1)
log_circ.cx(0,1)
log_circ.measure([0,1], [0,1])

backend = FakeTenerife()

transpile(log_circ,backend)

# Adjust log level for the transpiler
logging.getLogger('qiskit.transpiler').setLevel('INFO')
transpile(log_circ, backend)

# Setting up logging to deal with parallel execution (naieve way)
logging.getLogger('qiskit_transpiler').setLevel('DEBUG')
circuits = [log_circ, log_circ, log_circ]
transpile(circuits, backend)

# Format logging to deal with parallel execution better
formatter = logging.Formatter('%(name)s - %(processName)-10s - %(levelname)s: %(message)s')
handler = logging.getLogger().handlers[0]
handler.setFormatter(formatter)
transpile(circuits, backend)
