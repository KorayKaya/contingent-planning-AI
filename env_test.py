from environment import Map
from utilities import *
from agent import RandomAgent

env = Map.loadFromFile("./maps/bigmap.txt")
printMap(env)

# evaluate(RandomAgent(env), 1000)

input("Press Enter to exit...")
