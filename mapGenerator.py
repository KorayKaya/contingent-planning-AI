import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


"""
# MAP CHARACTERISTICS PARAMETERS
"""

_NUMBER_OF_NODES = 250
_LOBSTER_P_BACKBONE = 0.90
_LOBSTER_P_BEYOND_BACKBONE = 0.70
_MAP_DENSITY = 0.20
_NUMBER_OF_DISTRIBUTIONS = 20
_NUMBER_OF_EDGES = _NUMBER_OF_NODES * (1+_MAP_DENSITY)
_GENERATED_MAP_NUMBER = "LOBSTER-N-{N}-E-{E}-D-{D}".format(N=_NUMBER_OF_NODES,E=_NUMBER_OF_EDGES,D=_NUMBER_OF_DISTRIBUTIONS)


"""
#  DISTRIBUTION PARAMETERS
"""
# _DISTRIBUTION_TYPES = ["N","E"]
_DISTRIBUTION_TYPES = ["E"]
                                    
_MEAN = [100,150,200,250]
_SD = [5,7,10,12,15,20,25,30]
_DISTRIBUTIONS = ["{_DISTRIBUTIONS}({_MEAN},{_SD})".format( \
                                                            _DISTRIBUTIONS=_DISTRIBUTION_TYPES[np.random.randint(len(_DISTRIBUTION_TYPES))], \
                                                            _MEAN=_MEAN[np.random.randint(len(_MEAN))], \
                                                            _SD=_SD[np.random.randint(len(_SD))])\
                                                            for n in range(_NUMBER_OF_DISTRIBUTIONS)]


"""
# OUTPUT FILE PARAMETERS
"""
_MAP_TEXT_FILE = "generated_map_{_GENERATED_MAP_NUMBER}.txt".format(_GENERATED_MAP_NUMBER=_GENERATED_MAP_NUMBER)
_MAP_VISUAL_FILE = "generated_map_visuals_{_GENERATED_MAP_NUMBER}.png".format(_GENERATED_MAP_NUMBER=_GENERATED_MAP_NUMBER)
_GRAPH_PICKLE_FILE = "generated_map_pickle_{_GENERATED_MAP_NUMBER}.gpickle".format(_GENERATED_MAP_NUMBER=_GENERATED_MAP_NUMBER)

"""
# NX generates a random graph which is used to represent the map which the agents will then traverse.
""" 
# Graph = nx.gnm_random_graph(_NUMBER_OF_NODES, _NUMBER_OF_EDGES, directed=True)

"""
# Lobster graph
"""
Graph = nx.generators.random_graphs.random_lobster(_NUMBER_OF_NODES,_LOBSTER_P_BACKBONE,_LOBSTER_P_BEYOND_BACKBONE)

# while not nx.is_strongly_connected(Graph) or \
#     ( nx.has_path(Graph,source=0,target=_NUMBER_OF_NODES-1) and\
#     len(nx.shortest_path(Graph,source=0,target=_NUMBER_OF_NODES-1)) < 10):
#     Graph = nx.gnm_random_graph(_NUMBER_OF_NODES, _NUMBER_OF_EDGES, directed=True)

# Assign a distribution to each edge
for fr, to in Graph.edges():   
    _random_distribution = np.random.randint(1,_NUMBER_OF_DISTRIBUTIONS)
    Graph[fr][to]['Distribution'] = _random_distribution
    Graph[fr][to]['Distribution_string'] = "{_distribution}".format(_x=to,_distribution=_DISTRIBUTIONS[_random_distribution])


_nn = Graph.number_of_nodes()

# Generate output file
file = open("./maps/{_MAP_TEXT_FILE}".format(_MAP_TEXT_FILE=_MAP_TEXT_FILE),"w")


edges = {}
for fr, to in Graph.edges():
    # First side of the route
    if not fr in edges.keys():
        edges[fr] = str(to)
    else:
        edges[fr] = edges[fr] + " " + str(to)

    # Other side of the route
    if not to in edges.keys():
        edges[to] = str(fr)
    else:
        edges[to] = edges[to] + " " + str(fr)

edges = dict(sorted(edges.items()))

file.write("NODES" + "\n")
for edge in edges:
    words = edges[edge].split()
    words = [ "{_x}:{_distribution}".format(_x=x,_distribution=Graph[int(edge)][int(x)]['Distribution']) for x in words]
    words = [ x if not x[0] == "0" else "START{X_}".format(X_=x[1:]) for x in words]
    words = [ x if not x[:len(str(_nn-1))] == str(_nn-1) else "GOAL{X_}".format(X_=x[len(str(_nn)):]) for x in words]


    _outstr =  " ".join(words)
    if edge == 0:
        file.write("START" + " " + _outstr,)
    elif edge == _nn - 1:
        file.write("GOAL" + " " + _outstr)
    else:
        file.write(str(edge) + " " + _outstr)
    file.write("\n")
file.write("END NODES" + "\n")

# # file.write the adjacency list
# file.write("NODES" + "\n")
# for line in nx.generate_adjlist(Graph):

#     words = line.split()
#     words[1:] = [ "{_x}:{_distribution}".format(_x=x,_distribution=Graph[int(words[0])][int(x)]['Distribution']) for x in words[1:]]
#     words = [ x if not x[0] == "0" else "START{X_}".format(X_=x[1:]) for x in words]
#     words = [ x if not x[:len(str(_nn))] == str(_nn-1) else "GOAL{X_}".format(X_=x[len(str(_nn)):]) for x in words]
#     words[:1] = [ x if not x == str(_nn-1) else "GOAL" for x in words[:1]]
#     file.write(" ".join(words))
#     file.write("\n")
# file.write("END NODES" + "\n")

# for formatting purposes
file.write("\n\n")

# write to file the distributions
file.write("DISTS"+ "\n")
for n in range(_NUMBER_OF_DISTRIBUTIONS):
    file.write("{n} {_DISTRIBUTIONS}".format(n=n+1,_DISTRIBUTIONS=_DISTRIBUTIONS[n]))
    file.write("\n")
file.write("END DISTS"+"\n")
file.close()

# Plotting
f = plt.figure(frameon=False)
# pos = nx.circular_layout(Graph)
pos = nx.spring_layout(Graph)
# Generate Labels
labels = {}
for i in range(1,_nn-1):
    labels[i] = i
labels[0] = "START"
labels[_nn-1] = "GOAL"

#Plot nodes
nx.draw_networkx_labels(Graph, labels=labels, pos=pos, font_size=5, alpha=0.50)
nx.draw_networkx_nodes(Graph, pos, nodelist=[i for i in range(_nn)], alpha=.5, node_size=50, node_color='#669999')
nx.draw_networkx_nodes(Graph, pos, nodelist=[0], alpha=.5, node_size=50, node_color='#cc6699', label="START")
nx.draw_networkx_nodes(Graph, pos, nodelist=[_nn-1], alpha=.5, node_size=50, node_color="#cc6699", label="GOAL")

# Plot Edges
nx.draw_networkx_edges(Graph, pos, width=1.0, alpha=0.5, edge_color="black")

# Plot edge labels
edge_labels = {(u, v): w['Distribution_string'][0] for u,v,w in Graph.edges(data=True)}
# nx.draw_networkx_edge_labels(Graph, edge_labels = edge_labels, pos=pos, alpha=0.25, font_size=3)

# plt.show()
f.savefig("./graphs/{_MAP_VISUAL_FILE}".format(_MAP_VISUAL_FILE=_MAP_VISUAL_FILE), bbox_inches='tight', pad_inches=0, transparent=True)

# Store network object
nx.write_gpickle(Graph, "./pickle/{_GRAPH_PICKLE_FILE}".format(_GRAPH_PICKLE_FILE=_GRAPH_PICKLE_FILE))
