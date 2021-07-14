### Pauli Operators

from qiskit.opflow import I, X, Y, Z # Pauli operators
print(I,X,Y,Z)
print(1.5*I) # can carry a coefficient
print(2.5*X)
print(X+2.0*Y) # can be used in a sum
print(X ^ Y ^ Z) # tensor products use ^
print(X @ Y @ Z) # composition uses @

# complicated objects
print((X + Y) ^ (Y + Z)) # composing two sums
print((X + Y) ^ (Y + Z)) # tensoring two sums
print(I, X)
print(2.0 * X^Y^Z)
print(1.1 * ((1.2 * X)^(Y + (1.3 * Z))))


### Part 1: State Functions and Measurements


from qiskit.opflow import (StateFn, Zero, One, Plus, Minus, H, DictStateFn,
                           VectorStateFn, CircuitStateFn, OperatorStateFn)

print(Zero, One) # states |0> and |1>, respectively
print(Plus, Minus) # states |+> = 1/sqrt(2)*(|0> + |1>) and |-> = 1/sqrt(2)*(|0> - |1>), respectively

# the eval method returns the coefficients of hte 0 and 1 basis states
print(Zero.eval('0'))
print(Zero.eval('1'))
print(One.eval('1'))
print(Plus.eval('0'))
print(Minus.eval('1'))

# the adjoint method gives the dual vector of the quantum state (the bra of the ket, or vice versa)
print(One.adjoint())
print(~One)


## Algebraic operations and predicates


print((2.0 + 3.0j) * Zero) # construct (2 + 3i)|0>
print(Zero + One) # adding two DictStateFn returns an object of the same type

# you must normalize states by hand
import math
v_zero_one = (Zero + One) / math.sqrt(2)
print(v_zero_one)

# symbolic representation of a sum
print(Plus + Minus)

# composition operator used to perform an inner product
print(~One @ One)

# symbolic expressions may be evaluated with the eval method
print((~One @ One).eval())
print((~v_zero_one @ v_zero_one).eval())
print((~Minus @ One).eval())

# the composition operator @ is equivalent to calling the compose method
print((~One).compose(One))
assert (~One).compose(One) == ~One @ One

# inner products may also be computed using the eval method directly
print((~One).eval(One))

# symbolic tensor products are constructed as follows
print(Zero ^ Plus) # |0> + |+>
print((Zero ^ Plus).to_circuit_op()) # represented as a simple CircuitStateFn

# tensor powers can be constructed using the ^ operator
print(600 * ((One ^ 5) + (Zero ^ 5))) # 600(|11111> + |00000>)
print((One ^ Zero) ^ 3) #|10>^3

# the method to_matrix_op converts to VectorStateFn
print(((Plus^Minus)^2).to_matrix_op())
print(((Plus^One)^2).to_matrix_op())
print(((Plus^Minus)^2).to_matrix_op().sample())

# StateFn class serves as a factory and can take any applicable primitive in its constructor
print(StateFn({'0':1}))
print(StateFn({'0':1}) == Zero)
print(StateFn([0,1,1,0]))

from qiskit.circuit.library import RealAmplitudes
print(StateFn(RealAmplitudes(2)))



### Part 2: PrimitiveOp's



# the basic operators are sublclasses of PrimitiveOp
# like StateFn, PrimitiveOp is a factory and can take any applicable primitive in its constructor

from qiskit.opflow import X, Y, Z, I, CX, T, H, S, PrimitiveOp

## Matrix elements

print(X)
print(X.eval('0'))
print(X.eval('0').eval('1'))
print(CX)
print(CX.to_matrix().real) # remove imaginary part
print(CX.eval('01'))
print(CX.eval('01').eval('11'))

## Applying an operator to a state vector

print(X @ One) # X|1> = |0>
print((X @ One).eval())
print(X.eval(One))

# composition and tensor products of operators are effected with @ and ^
print(((~One^2) @ (CX.eval('01'))).eval())
print(((H^5) @ ((CX^2)^I) @ (I^CX^2))**2)
print((((H^5) @ ((CX^2)^I) @ (I^CX^2))**2) @ (Minus^5))
print(((H^I^I)@(X^I^I)@Zero))
print(~One @ Minus)



### Part 3: ListOp and subclasses



# ListOp is a container for effectivley vectorizing operations over a list of operators and states
from qiskit.opflow import ListOp

print((~ListOp([One, Zero]) @ ListOp([One, Zero])))

# distribute over lists using simplification method reduce
print((~ListOp([One, Zero]) @ ListOp([One, Zero])).reduce())

## OperatorStateFn

# construct observable corresponding to Pauli Z operator
print(StateFn(Z).adjoint())
StateFn(Z).adjoint()

# compute <0|Z|0>, <1|Z|1>, and <+|Z|+>
print(StateFn(Z).adjoint().eval(Zero))
print(StateFn(Z).adjoint().eval(One))
print(StateFn(Z).adjoint().eval(Plus))



### Part 4: Converters



# converters manipulate operators and states and perform building blocks of algorithms
import numpy as np
from qiskit.opflow import (I, X, Y, Z, H, CX, Zero, ListOp, PauliExpectation,
PauliTrotterEvolution, CircuitSampler, MatrixEvolution, Suzuki)
from qiskit.circuit import Parameter
from qiskit import Aer

## Evolutions, exp_i(), and the EvolvedOP

# express hamiltonian as linear combination of multi-qubit Pauli operators
two_qubit_H2 = (-1.0523732 * I^I) + \
                (0.39793742 * I^Z) + \
                (-0.3979374 * Z^I) + \
                (-0.0112801 * Z^Z) + \
                (0.18093119 * X^X)

print(two_qubit_H2)

# multiply hamiltonian by a Parameter
evo_time = Parameter('θ')
evolution_op = (evo_time*two_qubit_H2).exp_i()
print(evolution_op) # Note: EvolvedOPs print as exponentiations
print(repr(evolution_op))

# construct observable for Hamiltonian
h2_measurement = StateFn(two_qubit_H2).adjoint()
print(h2_measurement)

# construct bell state via CX(H tensor I)|00>
bell = CX @ (I ^ H) @ Zero
print(bell)

# evolve bell state
evo_and_meas = h2_measurement @ evolution_op @ bell
print(evo_and_meas)

# approximate exponentiation with two-qubit gates using PauliTrotterEvolution
trotterized_op = PauliTrotterEvolution(trotter_mode=Suzuki(order=2, reps=1)).convert(evo_and_meas)
print(trotterized_op)

# bind_parameters method traverses expression, binding values ot parameter names
bound = trotterized_op.bind_parameters({evo_time: .5})
bound[1].to_circuit().draw('mpl').show()



## Exppectations

# expectations are converters that enable the computation of expectation values and observables

# AerPauliExpectation converts an observable into a CircuitStateFn containin a special expectation snapshot
# which Aer can execute nativley with high performance
print(PauliExpectation(group_paulis=False).convert(h2_measurement))

# by default, group_paulis=True which uses the AbelianGrouper to convert the SummedOp into groups
# of mutually-qubit wise commuting Paulis, reducing circuit execution overhead
print(PauliExpectation().convert(h2_measurement))

# converters act recursivley, so we can convert our full evolution and measurement expression
diagonalized_meas_op = PauliExpectation().convert(trotterized_op)
print(diagonalized_meas_op)
evo_time_points = list(range(8))
h2_trotter_expectations = diagonalized_meas_op.bind_parameters({evo_time:evo_time_points})
print(h2_trotter_expectations.eval())

## Executing CircuitStateFn with the CircuitSampler

# CircuitSampler traverses an Operator and converts any CircuitStateFn into approximations of the
# resulting state function by a DictStateFn or VectorStateFn using a quantum backend.
sampler= CircuitSampler(backend=Aer.get_backend('aer_simulator'))
# sampler.quantum_instance.run_config.shots = 1000
sampled_trotter_exp_op = sampler.convert(h2_trotter_expectations)
sampled_trotter_energies = sampled_trotter_exp_op.eval()
print('Sampled Trotterized energies:\n{}'.format(np.real(sampled_trotter_energies)))

# note: circuits are replaced by dicts with square roots of the circuit sampling probablilities
print('Before:\n')
print(h2_trotter_expectations.reduce()[0][0])
print('After:\n')
print(sampled_trotter_exp_op.reduce()[0][0])
