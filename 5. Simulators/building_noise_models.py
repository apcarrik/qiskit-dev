import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Kraus, SuperOp
from qiskit.providers.aer import AerSimulator
from qiskit.tools.visualization import plot_histogram

# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import QuantumError, ReadoutError
from qiskit.providers.aer.noise import pauli_error
from qiskit.providers.aer.noise import depolarizing_error
from qiskit.providers.aer.noise import thermal_relaxation_error




### Qiskit Aer Noise Module




## Quantum Errors

# construct a 5% single-qubit Bit flip error
p_error = 0.05
bit_flip = pauli_error([('X', p_error), ('I', 1-p_error)])
phase_flip = pauli_error([('Z', p_error), ('I', 1-p_error)])
print(bit_flip)
print(phase_flip)

# compose two bit-flip and phase-flip errors
bitphase_flip = bit_flip.compose(phase_flip)
print(bitphase_flip)

# tensor product two bit-flip and phase-flip errors with bit-flip on qubit-0
# and phase-flip on qubit-1
error2 = phase_flip.tensor(bit_flip)
print(error2)

## Converting To and From QuantumChannel Operators

# convert to Kraus operator
bit_flip_kraus = Kraus(bit_flip)
print(bit_flip_kraus)

# convert to Superoperator
phase_flip_sop = SuperOp(phase_flip)
print(phase_flip_sop)

# convert back to a quantum error
print(QuantumError(bit_flip_kraus))

# check conversion is equivalent to original error
print("conversion is equivalent: ", QuantumError(bit_flip_kraus) == bit_flip)

## Readout Error

# measurement miss-assignment probabilities
p0given1 = 0.1
p1given0 = 0.05

roerror = ReadoutError([[1 - p1given0, p1given0], [p0given1, 1-p0given1]])
print(roerror)




### Adding Errors to a Noise Model




## All-Qubit Quantum Error

# create an empty noise model
noise_model = NoiseModel()

# add depolarizing error to all single qubit u1, u2, u3 gates
error = depolarizing_error(0.05, 1)
noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])

# print noise model info
print(noise_model)

## Specific Qubit Quantum Error

# create an empty noise model
noise_model = NoiseModel()

# add depolarizing error to all single qubit u1, u2, u3 gates (on qubit 0 only)
error = depolarizing_error(0.05, 1)
noise_model.add_quantum_error(error, ['u1', 'u2', 'u3'], [0])

# print noise model info
print(noise_model)

## Non-local Qubit Quantum Error

# create an empty noise model
noise_model = NoiseModel()

# add depolarizing error to all single qubit u1, u2, u3 gates (on qubit 0 only)
error = depolarizing_error(0.05, 1)
noise_model.add_nonlocal_quantum_error(error, ['u1', 'u2', 'u3'], [0], [2])

# print noise model info
print(noise_model)




### Noise Model Examples




# system specification
n_qubits = 4
circ = QuantumCircuit(n_qubits)

# test circuit
circ.h(0)
for qubit in range(n_qubits -1):
    circ.cx(qubit, qubit + 1)
circ.measure_all()
print(circ)

## Ideal Simulation

# ideal simulator and execution
sim_ideal = AerSimulator()
result_ideal = sim_ideal.run(circ).result()
plot_histogram(result_ideal.get_counts(0)).show()

## Noise Example 1: Basic Bit-flip Error Noise Model

# example error probabilities
p_reset = 0.03
p_meas = 0.1
p_gate1 = 0.05

# QuantumError objects
error_reset = pauli_error([('X', p_reset), ('I', 1-p_reset)])
error_meas = pauli_error([('X', p_meas), ('I', 1-p_meas)])
error_gate1 = pauli_error([('X', p_gate1), ('I', 1-p_gate1)])
error_gate2 = error_gate1.tensor(error_gate1)

# add errors to noise model
noise_bit_flip = NoiseModel()
noise_bit_flip.add_all_qubit_quantum_error(error_reset, "reset")
noise_bit_flip.add_all_qubit_quantum_error(error_meas, "measure")
noise_bit_flip.add_all_qubit_quantum_error(error_gate1, ["u1", "u2", "u3"])
noise_bit_flip.add_all_qubit_quantum_error(error_gate2, ["cx"])

print(noise_bit_flip)

## Execute Noisy Example 1

# create noisy simulator backend
sim_noise = AerSimulator(noise_model=noise_bit_flip)

# transpile circuit for noisy basis gates
circ_tnoise = transpile(circ, sim_noise)

# run and get counts
result_bit_flip = sim_noise.run(circ_tnoise).result()
counts_bit_flip = result_bit_flip.get_counts(0)

# plot noisy output
plot_histogram(counts_bit_flip).show()




### Noisy Example 2: T1/T2 Thermal Relaxation




# T1 and T2 values for qubits 0-3
T1s = np.random.normal(50e3, 10e3, 4) # sampled from normal distribution mean 50 microsec
T2s = np.random.normal(70e3, 10e3, 4) # sampled from normal distribution mean 50 microsec

# truncate random T2s <= T1s
T2s = np.array([min(T2s[j], 2* T1s[j]) for j in range(4)])

# instruction times (in ns)
time_u1 = 0 # virtual gate
time_u2 = 50 # single X90 pulse
time_u3 = 100 # two X90 pulses
time_cx = 300
time_reset = 1000 # 1 microsecond
time_measure = 1000 # 1 microsecond

# QuantumError objects
errors_reset = [thermal_relaxation_error(t1, t2, time_reset)
                for t1, t2 in zip(T1s, T2s)]
errors_measure = [thermal_relaxation_error(t1, t2, time_measure)
                for t1, t2 in zip(T1s, T2s)]
errors_u1 = [thermal_relaxation_error(t1, t2, time_u1)
                for t1, t2 in zip(T1s, T2s)]
errors_u2 = [thermal_relaxation_error(t1, t2, time_u2)
                for t1, t2 in zip(T1s, T2s)]
errors_u3 = [thermal_relaxation_error(t1, t2, time_u3)
                for t1, t2 in zip(T1s, T2s)]
errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(
             thermal_relaxation_error(t1b, t2b, time_cx))
                for t1a, t2a in zip(T1s, T2s)]
                for t1b, t2b in zip(T1s, T2s)]

# add errors to noise model
noise_thermal = NoiseModel()
for j in range(4):
    noise_thermal.add_quantum_error(errors_reset[j], "reset", [j])
    noise_thermal.add_quantum_error(errors_measure[j], "measure", [j])
    noise_thermal.add_quantum_error(errors_u1[j], "u1", [j])
    noise_thermal.add_quantum_error(errors_u2[j], "u2", [j])
    noise_thermal.add_quantum_error(errors_u3[j], "u3", [j])
    for k in range(4):
        noise_thermal.add_quantum_error(errors_cx[j][k], "cx", [j,k])

print(noise_thermal)


## Execute Noisy Example 2

# create noisy simulator backend
sim_thermal = AerSimulator(noise_model=noise_thermal)

# transpile circuit for noisy basis gates
circ_tthermal = transpile(circ, sim_thermal)

# run and get counts
result_thermal = sim_thermal.run(circ_tthermal).result()
counts_thermal = result_thermal.get_counts(0)

# plot noisy output
plot_histogram(counts_thermal).show()