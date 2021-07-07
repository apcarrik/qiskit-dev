import numpy as np
import networkx as nx

from qiskit import Aer
from qiskit.algorithms import NumPyMinimumEigensolver
from matplotlib import pyplot

num_nodes = 4
w = np.array([
    [0., 1., 1., 0.],
    [1., 0., 1., 0.],
    [1., 1., 0., 1.],
    [0., 1., 1., 0.],
])

G = nx.from_numpy_matrix(w)

layout = nx.random_layout(G, seed=10)
colors = ['r','g','b','y']
nx.draw(G,layout,node_color=colors)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos=layout, edge_labels=labels)
pyplot.show()



