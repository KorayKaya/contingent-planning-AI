from __future__ import annotations
from environment import Map
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import agent
from typing import Dict, List
from itertools import islice
from python_linq import From

# -----------------------------------------------------------------


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "

    "Source: https://stackoverflow.com/a/6822773"
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

# -----------------------------------------------------------------


def printMap(map: Map, path: List = ['START', 'GOAL'], doBlock: bool = False):
    """Print a figure of the map visualized as a networkx graph."""
    G = nx.DiGraph()
    G.add_nodes_from(map.nodes.keys())  # Add the nodes to the graph

    # Add the edges
    for fr, node in map.nodes.items():
        for to in node.edges.keys():
            G.add_edge(fr, to, weight=round(map.getEdgeCost(fr, to), 3))

    pos = nx.circular_layout(G)  # Generate the node positions for plotting
    # Generate weight labels for the edges
    labels = nx.get_edge_attributes(G, 'weight')
    # Generate the node colors
    node_colormap = []
    for node in G.nodes:
        if node in path:
            node_colormap.append('#cc6699')
        else:
            node_colormap.append('#669999')

    nx.draw(G, pos, with_labels=True, node_size=1200,
            font_size=10, node_color=node_colormap)

    # Set the edge weights in the plot
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, label_pos=0.6)
    plt.show(block=doBlock)

# -----------------------------------------------------------------


def getRandomPolicy(map: Map) -> Dict[str, str]:
    """
    Return a random dictionary where the keys are the states from a map
    and the values are one of their adjacent states.
    """
    actions = {}
    for state, node in map.nodes.items():
        actions[state] = random.choice(list(node.edges.keys()))

    return actions

# -----------------------------------------------------------------

def plotWithRunningMeanAndStd(scores: List, Nmean: int):
    """
    Plot a sequence of values with their running mean
    and the running mean +/- 2 times the deviation.
    """
    running_mean = []
    running_std = []

    for i in range(Nmean, len(scores)):
        running_mean.append(np.mean(scores[i-Nmean:i]))
        running_std.append(np.std(scores[i-Nmean:i]))
    
    x = np.arange(Nmean, len(scores))
    running_mean = np.array(running_mean)
    running_std = np.array(running_std)

    print()
    print("Mean cost", np.mean(scores))
    print("Cost STD", np.std(scores))
    print("Last", Nmean, "mean", running_mean[-1])
    print("Last", Nmean, "STD", running_std[-1])

    plt.figure("Costs")
    plt.plot(scores, label="Cost")
    plt.plot(x, running_mean, 'r', label="Running mean")
    plt.plot(x, running_mean + 2*running_std, 'r--', label="Running mean and two STD")
    plt.plot(x, running_mean - 2*running_std, 'r--')
    plt.legend()
    plt.xlabel("Days spent")
    plt.ylabel("Total cost")
    plt.show(block=False)

# -----------------------------------------------------------------


def evaluate(agent: agent.Agent, roundLimit: int = 10, printRoutes: bool = True, printQuantile: float = 0.9, Nmean: int = 10):
    """Evaluates and plots the performance of an agent after training it for several rounds."""
    rounds = 0
    routes = {}
    scores = []
    while rounds < roundLimit:
        
        agent.reset()
        score = 0.0
        rounds += 1

        route = ["START"]
        while agent.state != 'GOAL':
            route.append(agent.travel())
            score += agent.env.distributions[agent.env.nodes[route[-2]].edges[route[-1]].distribution].getObservation()
        
        route = " ".join(route)     # Make it string
        if route in routes:
            routes[route] += 1
        else:
            routes[route] = 1

        
        scores.append(score)

    print("=== FINAL ROUTE ===")
    print(route, "-", scores[-1])

    routes = (From(routes.items())
        .order(lambda x: x[1], descending=True)
        .select(lambda x: (x[0], x[1] * 1.0 / roundLimit))
        .toList()
    )
    print()
    print("=== TOP", printQuantile * 100, "% of all routes ===")
    s = 0.0
    for route in routes:
        s += route[1]
        print(route[0], "-", route[1])
        if s >= printQuantile:
            break
    
    plotWithRunningMeanAndStd(scores, Nmean)
    input("Press Enter to exit...")

# -----------------------------------------------------------------
