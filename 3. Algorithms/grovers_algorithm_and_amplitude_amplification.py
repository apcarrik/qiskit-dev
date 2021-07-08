from qiskit import QuantumCircuit
from qiskit.algorithms import AmplificationProblem


### Specify oracle for Grover's algorithm circuit


# the state we desire to find is '11'
good_state = ['11']

# specify the oracle that marks the state '11' as a good solution
oracle = QuantumCircuit(2)
oracle.cz(0, 1)

# define Grover's algorithm
problem = AmplificationProblem(oracle, is_good_state=good_state)

# now we can have a look at the Grover operator aht is used in running the algorithm
problem.grover_operator.draw(output='mpl').show()


### Specify backend and call run method of Grover with backend to execute circuits


from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import Grover

aer_simulator = Aer.get_backend('aer_simulator')
grover = Grover(quantum_instance=aer_simulator)
result = grover.amplify(problem)
print('Using QuantumCircuit as oracle')
print('Result type:', type(result))
print('Success!' if result.oracle_evaluation else 'Failure')
print(f'Top measurement: {result.top_measurement}')


### Using different types of classes as oracle for Grover's algorithm


# using statevector as oracle
from qiskit.quantum_info import Statevector
oracle = Statevector.from_label('11')
problem = AmplificationProblem(oracle, is_good_state=['11'])

grover = Grover(quantum_instance=aer_simulator)
result = grover.amplify(problem)
print('\nUsing Statevector as oracle')
print('Result type:', type(result))
print('Success!' if result.oracle_evaluation else 'Failure')
print(f'Top measurement: {result.top_measurement}')
problem.grover_operator.oracle.draw(output='mpl').show()

# using phase oracle
from qiskit.circuit.library.phase_oracle import PhaseOracle
from qiskit.exceptions import MissingOptionalLibraryError

# `Oracle` (`PhaseOracle`) as the `oracle` argument
expression = '(a & b)'
try:
    oracle = PhaseOracle(expression)
    problem = AmplificationProblem(oracle)
    problem.grover_operator.oracle.draw(output='mpl').show()
except MissingOptionalLibraryError as ex:
    print(ex)



### Amplitude Amplification


# specifying `state preparation` to prepare a superposition of |01>, |10>, and |11>
import numpy as np

oracle = QuantumCircuit(3)
oracle.h(2)
oracle.ccx(0,1,2)
oracle.h(2)

theta = 2 * np.arccos(1 / np.sqrt(3))
state_preparation = QuantumCircuit(3)
state_preparation.ry(theta, 0)
state_preparation.ch(0,1)
state_preparation.x(1)
state_preparation.h(2)

# we only care about the first two bits beign in state 1, thus add both possiblities for the last qubit
problem = AmplificationProblem(oracle, state_preparation=state_preparation,
                               is_good_state=['110','111'])

# state preperation
print('Printing state preperation circuit')
problem.grover_operator.state_preparation.draw(output='mpl').show()

# run circuit
grover = Grover(quantum_instance=aer_simulator)
result = grover.amplify(problem)
print('Success!' if result.oracle_evaluation else 'Failure :(')
print(f'Top measurement: {result.top_measurement}')



### Full Flexibility


# set good state as |111> but with 5 qubits
from qiskit.circuit.library import GroverOperator, ZGate

oracle = QuantumCircuit(5)
oracle.append(ZGate().control(2), [0,1,2])
oracle.draw(output='mpl').show()

# Grover operator implements zero reflection on all 5 qubits, as default
grover_op = GroverOperator(oracle, insert_barriers=True)
grover_op.draw(output='mpl').show()

# since we only need to consider first 3 qubits, set that as parameter to GroverOperator
grover_op = GroverOperator(oracle, reflection_qubits=[0,1,2], insert_barriers=True)
grover_op.draw(output='mpl').show()



### Specify good_state in different ways


# list of binary strings good state
oracle = QuantumCircuit(2)
oracle.cz(0,1)
good_state = ['11', '00']
problem = AmplificationProblem(oracle, is_good_state=good_state)
print('Binary strings good state')
print(f'State 11 is a good state: {problem.is_good_state("11")}')

# list of integers good state
oracle = QuantumCircuit(2)
oracle.cz(0,1)
good_state = [0, 1]
problem = AmplificationProblem(oracle, is_good_state=good_state)
print('Integer good state')
print(f'State 11 is a good state: {problem.is_good_state("11")}')

# `Statevector` good state
from qiskit.quantum_info import Statevector

oracle = QuantumCircuit(2)
oracle.cz(0,1)
good_state = Statevector.from_label('11')
problem = AmplificationProblem(oracle, is_good_state=good_state)
print('Statevector good state')
print(f'State 11 is a good state: {problem.is_good_state("11")}')

# Callable good state
def callable_good_state(bitstr):
    if bitstr == '11':
        return True
    return False

oracle = QuantumCircuit(2)
oracle.cz(0,1)
problem = AmplificationProblem(oracle, is_good_state=callable_good_state)
print('Callable good state')
print(f'State 11 is a good state: {problem.is_good_state("11")}')


### Change number of iterations


# integer iteration
oracle = QuantumCircuit(2)
oracle.cz(0, 1)
problem = AmplificationProblem(oracle, is_good_state=['11'])
grover = Grover(iterations=1)

# list iteration
oracle = QuantumCircuit(2)
oracle.cz(0, 1)
problem = AmplificationProblem(oracle, is_good_state=['11'])
grover = Grover(iterations=[1,2,3])


# using sample_from_iterations
oracle = QuantumCircuit(2)
oracle.cz(0, 1)
problem = AmplificationProblem(oracle, is_good_state=['11'])
grover = Grover(iterations=[1,2,3], sample_from_iterations=True)

# use optimal_num_iterations when number of solutions is known
iterations = Grover.optimal_num_iterations(num_solutions=1, num_qubits=8)
print(f'Optimal number of iterations: {iterations}')



### Apply post processing


# convert bit-representation of measurement to DIMACS CNF format
def to_DIMACS_CNF_format(bit_rep):
    return [index+1 if val==1 else -1 * (index+1) for index, val in enumerate(bit_rep)]

oracle = QuantumCircuit(2)
oracle.cz(0, 1)
problem = AmplificationProblem(oracle, is_good_state=['11'],
                               post_processing = to_DIMACS_CNF_format)
print("DIMACS CNF format: ", problem.post_processing([1, 0, 1]))

