from environment import Map
import utilities as util
from typing import List
import random
import numpy as np
import matplotlib.pyplot as plt

class Genetic:
    _invalidEdgePenalty = 10000
    _population_size = 30
    _crossover_prob = 0.9
    _mutation_prob = 0.3
    _nr_of_parents = 5
    _crossoverIter = 10

    # -----------------------------------------------------------------

    def __init__(self, map: Map, isVerbose: bool = False):
        self.map: Map = map
        self.nr_of_nodes: int = len(map.nodes)
        self.node_set: set = map.nodes.keys() - {"START", "GOAL"}
        self.population: List = []
        self.selected_parents = []

    # -----------------------------------------------------------------

    def init_population(self):
        """Initialize the population with random (valid) paths from START to GOAL."""
        for _ in range(self._population_size):
            chromosome = []
            current_node = 'START'
            
            adj_nodes = set(self.map.nodes[current_node].edges.keys()) - set(chromosome)

            while len(adj_nodes) != 0 and 'GOAL' not in adj_nodes:
                chromosome.append(current_node)
                current_node = random.sample(adj_nodes, 1)[0]
                adj_nodes = set(self.map.nodes[current_node].edges.keys()) - set(chromosome)

            chromosome.append(current_node)
            chromosome.append('GOAL')

            self.population.append(chromosome)

    # -----------------------------------------------------------------

    def fitness(self, chromosome: List):
        """Return the inverse of the total cost of a path."""
        ret = 0.0
        for edge in util.window(chromosome, 2):
            try:
                ret = ret + self.map.getEdgeCost(edge[0], edge[1])
            except KeyError:
                ret = ret + self._invalidEdgePenalty

        return 1/ret

    # -----------------------------------------------------------------

    @staticmethod
    def find_matching_indices(list1, list2):
        """Return the indices of matching elements as a list of pairs."""
        inverse_index = {element: index for index, element in enumerate(list1)}
        return [(inverse_index[element], index) for index, element in enumerate(list2) if element in inverse_index]

    # -----------------------------------------------------------------

    @staticmethod
    def is_there_matching(list1, list2):
        """Return true if there are matching elements in the two lists, false otherwise."""
        return any(i in list1 for i in list2)

    # -----------------------------------------------------------------

    @staticmethod
    def mate(parent1: List, parent2: List):
        """Perform the actual crossover operation."""
        children = []
        matching = Genetic.find_matching_indices(parent1, parent2)
        if len(matching) != 0:
            parent1_ind, parent2_ind = random.choice(matching)
            if not Genetic.is_there_matching(parent1[:parent1_ind], parent2[parent2_ind:]):
                child1 = parent1[:parent1_ind] + parent2[parent2_ind:]
                children.append(child1)
            if not Genetic.is_there_matching(parent2[:parent2_ind], parent1[parent1_ind:]):
                child2 = parent2[:parent2_ind] + parent1[parent1_ind:]
                children.append(child2)
        return children

    # -----------------------------------------------------------------

    def crossover(self):
        """Create new individuals by applying crossover to two random parents repeatedly."""
        for _ in range(self._crossoverIter):
            if random.random() <= self._crossover_prob:
                parent1 = random.choice(self.selected_parents)
                parent2 = random.choice(self.selected_parents)
                self.population.extend(Genetic.mate(parent1, parent2))

    # -----------------------------------------------------------------

    def selection(self):
        """Choose _nr_of_parents chromosomes for crossover using Roulette Wheel Selection."""
        probabilities = [self.fitness(x) for x in self.population]
        probabilities = [x / sum(probabilities) for x in probabilities]
        selected_indices = np.random.choice(len(self.population), 
                                            size=self._nr_of_parents, p=probabilities, replace=False)
        self.selected_parents = [self.population[i] for i in selected_indices]

    # -----------------------------------------------------------------

    def mutate(self):
        """Change a random node to a new one in a path."""
        for chrom in self.population[self._nr_of_parents:]:
            if random.random() <= self._mutation_prob:
                # Exclude START and GOAL
                ind = random.randint(1, len(chrom) - 2)
                if (len(self.node_set - (set(chrom))) >= 1):
                    chrom[ind] = random.sample(
                        self.node_set - set(chrom), 1)[0]

    # -----------------------------------------------------------------

    def sort_population(self):
        """Sort the population in an ascending order using the fitness function."""
        self.population = sorted(
            self.population, key=lambda x: self.fitness(x), reverse=True)

    # -----------------------------------------------------------------

    def get_first_fitness(self):
        """Return the current best path."""
        return self.fitness(self.population[0])

    # -----------------------------------------------------------------

    def prune_population(self):
        """Remove the excess children from the population."""
        self.population = self.population[:self._population_size]

# -----------------------------------------------------------------

# Select the map here: tunnelbana and and the other lobsters are not very
# useful because we find the optimal path during initialization
env = Map.loadFromFile("./maps/generated_map_LOBSTER-N100.txt")
# env = Map.loadFromFile("./maps/bigmap.txt")
# env = Map.loadFromFile("./maps/map3.txt") #tunnelbana
scores = []
max_gen = 300

ga = Genetic(env)
ga.init_population()
ga.sort_population()

gen = 0
while gen < max_gen:
    scores.append(1/ga.get_first_fitness())

    ga.selection()
    ga.crossover()
    ga.mutate()

    ga.sort_population()
    ga.prune_population()
    ga.map.reset()
    gen += 1

print("Mean cost", np.mean(scores))
print("Cost STD", np.std(scores))
# util.printMap(env, path=ga.population[0], doBlock=False)
util.plotWithRunningMeanAndStd(scores, Nmean=10)
input("Press Enter to exit...")
