from __future__ import annotations
from environment import Map, Node, Edge
from abc import abstractmethod
from queue import Queue
from typing import Dict
from python_linq import From
from random import random, choice
import utilities
import numpy as np


class Agent:
    """The abstract base class for agents that move around the map."""

    def __init__(self, env: Map, isVerbose: bool=False):
        self.env = env
        self.state: str = 'START'       # Always start at START
        self.score: float = 0.0
        self.verbose = isVerbose #TODO add verbosity option

    @abstractmethod
    def travel(self) -> str:
       pass

    def reset(self):
        """ Resets the agent and environment """
        self.state = 'START'
        self.score = 0.0
        self.env.reset()


class RandomAgent(Agent):
    """An agent that moves randomly until the goal is reached."""

    def __init__(self, env: Map, isVerbose: bool=False):
        super().__init__(env, isVerbose)
        self.policy: Dict[str, str] = utilities.getRandomPolicy(env)
    
    def travel(self) -> str:
        """Return where the agent chooses to go from a given state."""
        if self.state == 'GOAL':
            return 'GOAL'
        elif self.state in self.policy:
            self.score += self.env.getEdgeCost(self.state, self.policy[self.state])
            self.state = self.policy[self.state]
            return self.state
        else:
            raise Exception('Action not found for ' + self.state + ' in the policy!') 


class MDPAgent(Agent):

    def __init__(self, env: Map, use_estimates: bool=True, replan: bool=True, observe_future: bool=True, isVerbose: bool=False):
        """
            use_estimates: Whether or not to use the estimated mean and variance or the true ones
            replan: Replan at every node
            observe_future: observe the cost of edges connected to the current state (i.e. look ahead)
        """
        super(MDPAgent,self).__init__(env, isVerbose)
        
        self.replan: bool = replan
        self.use_estimates: bool = use_estimates
        self.observe_future = observe_future

        self.states: Dict[str, float] = {}      # Value of each state
        self.plan()
    
    def reset(self):
        super(MDPAgent,self).reset()
        self.plan()

    def getCost(self, fr, to):
        """Gets cost that agent is allowed to see"""

        distribution = self.env.getEdgeDistribution(fr, to)
        if distribution.isObserved() and self.replan:   
            return distribution.getObservation()       # If observed cost, return the true cost
        elif self.use_estimates:      
            return distribution.getEstimatedMean()     # If unobserved, return estimated mean
        else:
            return distribution.getMean()              # If using true mean, return that

    def plan(self):
        """ Calculates the value of each state """
        
        self.states: Dict[str, float] = {}  # Reset values
        self.states["GOAL"] = 0         # Initialize goal state

        queue = Queue()     # Set L
        for node in self.env.nodes["GOAL"].parents:
            value = - self.getCost(node, "GOAL")      # Get cost from node to GOAL
            queue.put((node, value))

        while not queue.empty():        
            
            node, value = queue.get()           # Get s', v'

            if node in self.states and value <= self.states[node]:
                continue                        # We found a new route from s' to goal but its total reward is smaller than the best found one. Stop
            
            self.states[node] = value       # Save V(s') = v'

            for parent in self.env.nodes[node].parents:
                queue.put((parent, value - self.getCost(parent, node)))     # add previous node with value v = v' + r (reward = - cost)

    def travel(self):

        if self.replan:
            self.plan()

        bestState = From(self.env.nodes[self.state].edges).argmax(lambda key: - self.getCost(self.state, key) + self.states[key])     # Find the edge that minimizes r + v'

        # self.score += self.env.getEdgeCost(self.state,bestState)
        self.state = bestState

        # Observe costs of nearby edges
        if self.observe_future:
            for edge in self.env.nodes[self.state].edges:
                self.env.getEdgeCost(self.state, edge)

        return self.state

class QAgent(Agent):
    """
        Extension of Agent using Q-learning.
    """
    def __init__(self, env: Map, config: QAgentConfig, isVerbose=False):
        super().__init__(env, isVerbose=isVerbose)
        
        self.config = config

        self.qValues: Dict[str, Dict[str, QAgent.EdgeData]] = {}    # qValues[s][a].Q returns Q-value
        
        for node in self.env.nodes.values():        # Initialize
            nodeQValues = {}
            for edge in node.edges:
                nodeQValues[edge] = QAgent.EdgeData(config)
            self.qValues[node.name] = nodeQValues

    def travel(self):
        
        # Find greedy action
        bestAction = From(self.env.nodes[self.state].edges).argmax(lambda key: self.qValues[self.state][key].Q)

        # Pick random action w.p. eps
        bestAction = bestAction if random() >= self.config.epsilon else choice(list(self.env.nodes[self.state].edges.keys()))

        # Find max Q value of next state
        if bestAction == "GOAL":
            q_p = 0
        else:
            q_p = From(self.qValues[bestAction].values()).select(lambda edge: edge.Q).max()
        
        cost = self.env.getEdgeCost(self.state, bestAction)
        self.qValues[self.state][bestAction].update(cost, q_p)  # Training happens in here!

        # self.score += cost
        self.state = bestAction
        return self.state

    class EdgeData:
        def __init__(self, config: QAgentConfig):
            
            self._config = config
            self._num_of_observations = 0
            
            self.Q = self._config.initialQ
            self.learning_rate = self._config.learning_rate
            
        def update(self, cost: float, bestNextEdgeQ: float):
            """Trains the Q-value and estimates of distribution"""
            self._num_of_observations += 1
            alpha = self.learning_rate / np.sqrt(self._num_of_observations)

            reward = - cost
            self.Q += alpha * ( reward + self._config.discount * bestNextEdgeQ - self.Q )  # Update formula!

class QAgentConfig:
    def __init__(self, initialQ: float, learning_rate: float, variance_weight: float, discount: float, epsilon: float):
        self.initialQ = initialQ
        self.learning_rate = learning_rate
        self.variance_weight = variance_weight   # unused atm
        self.discount = discount
        self.epsilon = epsilon