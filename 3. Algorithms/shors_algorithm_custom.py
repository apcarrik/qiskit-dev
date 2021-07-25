import matplotlib as plt
import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram
from math import gcd
from numpy.random import randint
import pandas as pd
from fractions import Fraction
print("Imports Successful")


### Problem: Period Finding

# Builds circuit for operator U such that U|y> = |ay mod 15>
def c_amod15(a, power):
    """Controlled multiplication by a mod 15"""
    if a not in [2,7,8,11,13]:
        raise ValueError("'a' must be 2,7,8,11, or 13")
    U = QuantumCircuit(4)
    for iteration in range(power):
        if a in [2,13]:
            U.swap(0,1)
            U.swap(1,2)
            U.swap(2,3)
        if a in [7,8]:
            U.swap(2,3)
            U.swap(1,2)
            U.swap(0,1)
        if a == 11:
            U.swap(1,3)
            U.swap(0,2)
        if a in [7,11,13]:
            for q in range(4):
                U.x(q)
    U = U.to_gate()
    U.name = "%i^%i mod 15" % (a, power)
    c_U = U.control()
    return c_U

# Specify variables
n_count = 8 # number of counting qubits
a = 7

# Import QFT circuit
def qft_dagger(n):
    """n-qubit QFTdagger the first n qubits in circuit"""
    qc = QuantumCircuit(n)
    # Don't forget the swaps!
    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi/float(2**(j-m)), m, j)
        qc.h(j)
    qc.name = "QFTdagger"
    return qc

# Create QuantumCircuit with n_count counting qubits plus 4 qubits for U to act on
qc = QuantumCircuit(n_count + 4, n_count)

# Initialize counting qubits in state |+>
for q in range(n_count):
    qc.h(q)

# Add auxiliary register in state |1>
qc.x(3+n_count)

# Do controlled-U operations
for q in range(n_count):
    qc.append(c_amod15(a, 2**q), [q] + [i+n_count for i in range(4)])

# Do inverse-QFT
qc.append(qft_dagger(n_count), range(n_count))

# Measure circuit
qc.measure(range(n_count), range(n_count))
qc.draw(output='mpl', fold=-1).show() # -1 means do not fold

# Show histogram results
aer_sim = Aer.get_backend('aer_simulator')
t_qc = transpile(qc, aer_sim)
qobj = assemble(t_qc)
results = aer_sim.run(qobj).result()
counts = results.get_counts()
plot_histogram(counts).show()

# Show text results
rows, measured_phases = [], []
for output in counts:
    decimal = int(output, 2) # convert (base 2) string to decimal
    phase = decimal/(2**n_count) # find corresponding eigenvalue
    measured_phases.append(phase)
    rows.append(
        [f"{output}(bin) = {decimal:>3}(dec)",
         f"{decimal}/{2**n_count} = {phase:.2f}"]
    )# add these valeus to the rows in our table
headers=["Register Output", "Phase"]
df = pd.DataFrame(rows, columns=headers)
print(df)

# Find s and r using continuded fractions algorithm
rows = []
for phase in measured_phases:
    frac = Fraction(phase).limit_denominator(15)
    rows.append([phase, f"{frac.numerator}/{frac.denominator}", frac.denominator])
headers = ["Phase", "Fraction", "Guess for r"]
df = pd.DataFrame(rows, columns= headers)
print(df)


### Modular Exponentiation


# Use repeated squaring to calcualte exponential a^(2^j) mod N
def a2jmodN(a, j, N):
    """Compute a^(2^j) (mod N) by repeated squaring"""
    for i in range(j):
        a = np.mod(a**2, N)
    return a

print(a2jmodN(7,2049, 53))


### Factoring and Period Finding


# Assume N is the product of two primes
N = 15
np.random.seed(1)
a = randint(2,15)
print("a=",a)

# Check a isn't already a non-trivial factor of N
from math import gcd
assert gcd(a,N) == 1

# Apply Shor's order finding algorithm for N=15
def qpe_amod15(a):
    n_count = 8
    qc = QuantumCircuit(4+n_count, n_count)
    for q in range(n_count):
        qc.h(q) # initialize counting qubits in state |+>
    qc.x(3+n_count) # set auxiliary register in state |1>
    for q in range(n_count): # do controlled-U operations
        qc.append(c_amod15(a, 2**q),
                  [q] + [i+n_count for i in range(4)])
    qc.append(qft_dagger(n_count), range(n_count)) # Do inverse-QFT
    qc.measure(range(n_count), range(n_count))

    # Simulate results
    aer_sim = Aer.get_backend('aer_simulator')
    t_qc = transpile(qc, aer_sim)
    obj = assemble(t_qc, shots=1)
    result = aer_sim.run(qobj, memory=True).result() # setting memory=True allows us to see list of each sequential reading
    readings = result.get_memory()
    print("Register Reading: ", readings[0])
    phase = int(readings[0],2)/(2**n_count)
    print("Corresponding Phase: %f" % phase)
    return phase

# Find guess for r given phase
phase = qpe_amod15(a) # Phase = s/r
Fraction(phase).limit_denominator(15) # Denominator should tell us r

# Get fraction
frac = Fraction(phase).limit_denominator(15)
s, r = frac.numerator, frac.denominator
print(r)

# use r to guess factors of N
guesses = [gcd(a**(r//2)-1, N), gcd(a**(r//2)+1, N)]
print("guesses:", guesses)

# repeat guess algorithm until at least one factor of 15 is found
a = 7
factor_found = False
attempt = 0
while not factor_found:
    attempt += 1
    print("\nAttempt %i:" % attempt)
    phase = qpe_amod15(a) # Phase = s/r
    frac = Fraction(phase).limit_denominator(N) # Denominator should tell us r
    r = frac.denominator
    print("Result: r = %i" % r)
    if phase != 0:
        # Guesses for factors are gcd(x^(r/2) +/- 1 , 15)
        guesses = [gcd(a**(r//2)-1, N), gcd(a**(r//2)+1, N)]
        print("Guessed Factors: %i and %i" % (guesses[0], guesses[1]))
        for guess in guesses:
            if guess not in [1,N] and (N % guess) == 0: # check to see if guess is factor
                print("*** Non-trivial factor found: %i ***" % guess)
                factor_found = True

