from __future__ import annotations
from typing import List, Dict
from distributions import Distribution
from python_linq import From

class Map:

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.distributions: Dict[str, Distribution] = {}

    def getEdgeDistribution(self, fr: str, to: str) -> Distribution:
        """Returns the distribution of the edge between fr and to."""
        return self.distributions[self.nodes[fr].edges[to].distribution]

    def getEdgeCost(self, fr: str, to: str) -> float:                
        """Returns the actual cost of the edge."""
        return self.getEdgeDistribution(fr, to).getObservation()
    
    def reset(self):                                                        
        """Resets the environment for a new run, i.e. resets all observed true costs."""
        for _, distribution in self.distributions.items():
            distribution.reset()

    @staticmethod
    def loadFromFile(filename: str) -> Map:                                 
        """Static method to load a map from file."""
        re = Map()

        with open(filename, mode="r", encoding="utf-8") as file:
            lines = file.read().splitlines()
        
        i = 0
        while i < len(lines):
            if lines[i] == "NODES":
                # Found start of NODES, let's find the end
                j = i + 1
                while lines[i] != "END NODES":
                    i += 1
                # Add nodes found
                re._addNodes(lines, j, i)
            
            elif lines[i] == "DISTS":
                # Found start of DISTS, let's find the end
                j = i + 1
                while lines[i] != "END DISTS":
                    i += 1
                # Add nodes found
                re._addDists(lines, j, i)
            else:
                i += 1

        return re

    def _addNodes(self, lines: List[str], fr: int, to: int):                                    
        """Adds the nodes to the map during loading."""
        self.nodes = From(lines).skip(fr).take(to - fr).select(Node).toDict(key=lambda node: node.name)
        for node in self.nodes.values():
            for edge in node.edges:
                self.nodes[edge].parents.append(node.name)


    def _addDists(self, lines: List[str], fr: int, to: int):                                    
        """Adds the distributions to the map during loading."""
        self.distributions = From(lines).skip(fr).take(to - fr).select(Distribution.getDistribution).toDict(key=lambda dist: dist.name)

class Node:
    def __init__(self, in_str: str):
        splits = in_str.split(' ')
        
        self.name: str = splits[0]
        self.edges: Dict[str, Edge] = From(splits[1:]).select(Edge).toDict(key=lambda edge: edge.target)
        self.parents: List[str] = []

class Edge:
    def __init__(self, in_str: str):
        splits = in_str.split(':')
        self.target: str = splits[0]
        self.distribution: str = splits[1]
