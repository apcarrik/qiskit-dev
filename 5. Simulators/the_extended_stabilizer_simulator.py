from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.tools.visualization import plot_histogram

import random

circ = QuantumCircuit(40,40)

# initialize with a Hadamard layer
circ.h(range(40))

# apply some random CNOT and T gates
qubit_indicies = [i for i in range(40)]
for i in range(10):
    control, target, t = random.sample(qubit_indicies,3)
    circ.cx(control,target)
    circ.t(t)
circ.measure(range(40), range(40))

# create statevector simulator
statevector_simulator = AerSimulator(method='statevector')

# transpile circuit for backend
tcirc = transpile(circ, statevector_simulator)

# try and run circuit
statevector_result = statevector_simulator.run(tcirc, shots=1).result()
print('This succeeded?: {}'.format(statevector_result.success))
print('Why not?: {}'.format(statevector_result.status))

# create an extended stablilizer method simulator
extended_stabilizer_simulator = AerSimulator(method='extended_stabilizer')

# transpile circuit for backend
tcirc = transpile(circ, extended_stabilizer_simulator)

# run circuit and get results
extended_stabilizer_result = extended_stabilizer_simulator.run(tcirc, shots=1).result()
print('This succeeded?: {}'.format(extended_stabilizer_result.success))



### Extended Stabilizer Only Gives Approximate Results



small_circ = QuantumCircuit(2,2)
small_circ.h(0)
small_circ.cx(0,1)
small_circ.t(0)
small_circ.measure([0,1],[0,1])

# the circuit should give 00 and 11 with equal probability
expected_results = {'00': 50, '11':50}
tsmall_circuit = transpile(small_circ, extended_stabilizer_simulator)
result = extended_stabilizer_simulator.run(
    tsmall_circuit, shots=100).result()
counts = result.get_counts(0)
print('100 shots in {}s'.format(result.time_taken))
plot_histogram([expected_results, counts], legend=["Expected", "Extened Stabilizer"]).show()

## Add Runtime Options for Extended Stabilizer Simulator

opts = {"extended_stabilizer_approximation_error":0.03}
reduced_error_result = extended_stabilizer_simulator.run(
    tsmall_circuit, shots=100, **opts).result()
reduced_error_counts = reduced_error_result.get_counts(0)
print('100 shots in {}s'.format(reduced_error_result.time_taken))
plot_histogram([expected_results, reduced_error_counts],
               legend=["Expected", "Extened Stabilizer"]).show()

## Simulator Options

# if you expect your output will be concentrated on a few states, you can optimize the
# simulations by reducing the extended_stabilizer_simulator_mixing_time option
print("The circuit above, with 100 shots and precision 0.03 "
      "and default mixing time, needed {}s".format(int(reduced_error_result.time_taken)))
opts = {"extended_stabilizer_approximation_error": 0.03,
        "extended_stabilizer_simulator_mixing_time": 100}
optimized_result = extended_stabilizer_simulator.run(
    tsmall_circuit, shots=100, **opts).result()
print('Dialing down the mixing time, the circuit completed in just {}s'.format(
    optimized_result.time_taken))

# if your circuit has non-zero probability on all amplitudes (e.g. it is a random circuit),
# then you can avoid this expensive re-mixing step to take multiple shots from the output
# at once.
opts = {"extended_stabilizer_simulator_mixing_time": 100}

multishot = extended_stabilizer_simulator.run(
    tcirc, shots=100, **opts).result()
print("100 shots took {}s".format(multishot.time_taken))

opts = {
    'extended_stabilizer_measure_sampling': True,
    'extended_stabilizer_mixing_time': 100
}

measure_sampling = extended_stabilizer_simulator.run(
    tcirc, shots=100, **opts).result()
print("With the optimization, 100 shots took {}s".format(measure_sampling.time_taken))




