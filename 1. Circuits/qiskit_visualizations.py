from qiskit import *
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
from matplotlib import pyplot

# Quanum ciruit to make a Bell state
bell = QuantumCircuit(2,2)
bell.h(0)
bell.cx(0,1)

meas = QuantumCircuit(2,2)
meas.measure([0,1],[0,1])

# execute quantum circuit
backend = BasicAer.get_backend('qasm_simulator')
circ = bell + meas
result = execute(circ,backend,shots=1000).result()
counts = result.get_counts(circ)
print(counts)

plot_histogram(counts).show() # don't need pyplot.show() anymore
# could also call .savefig('output.png') to save to file

second_result = execute(circ,backend, shots=1000).result()
second_counts = second_result.get_counts(circ)
legend = ['first execution', 'second execution']
plot_histogram([counts, second_counts], legend=legend, sort='desc', figsize=(15,12),
               color=['orange', 'black'], bar_labels=False).savefig('histogram1.png')

### Advanced Visualizations
