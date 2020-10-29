Planning algorithms
===================

We have implemented essentially three different algorithms, that can be run in various ways. There are also a number of different maps.

Requirements
------------

To run the program, python >= 3.7.0 is required (3.6 will not work due to the typing that we have used). Additionally, the packages in `requirements.txt` need to be installed, e.g. through the command
```
pip3 install -r requirements.txt
```

Maps
----

The maps can be found in `./maps/` and are made up of a list of nodes and a list of distributions. Each node is followed by multiple `NODE:DISTR` pairs which describe the nodes to which one can proceed to and what distribution their edge costs are given by. Each distribution can be either a shifted exponential defined by shift and scale (as numpy expects), or a Gaussian defined by mean and variance.

Agents
------

All agents inherit from the base class `Agent`, and they are all located in `agent.py`. The evaluation essentially takes an instance of the base class and runs it using the abstract method `agent.travel`. The evaluation function is found in `utilities.py`.

### MDP

The MDP algorithm can be run through the file `MDP.py`,
```
python3.7 MDP.py
```
Inside that file, one can choose map and choose which version is run. The algorithm used in the report use all flags as `True`. By choosing them all to `False`, the algorithm is essentially a dynamic programming optimization procedure that returns the optimal path.

### Q-Learning

The Q-Learning can be run through the file `QAgent.py`,
```
python3.7 QAgent.py
```
There are a number of parameters to tune to alter the behavior in that file, as well as which map to run.

### Genetic algorithm

The genetic algorithm can be run through the file `genetic.py`.
```
python3.7 genetic.py
```

Both the implementation and the driver code for the genetic algorithm can be found in genetic.py. The GA parameters (population size, mutation/crossover probabilites etc.) are defined as class attributes, while the max_generation parameter can be set in the driver code.

The used map can be printed by uncommenting the printMap() call at the bottom of the file.

