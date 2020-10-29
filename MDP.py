from environment import Map
from utilities import evaluate
from agent import MDPAgent

# env = Map.loadFromFile("./maps/map1.txt")
# env = Map.loadFromFile("./maps/bigmap.txt")
# env = Map.loadFromFile("./maps/generated_map_N=5.E=5.D=20.txt")
# env = Map.loadFromFile("./maps/generated_map_N=25.E=31.D=20.txt")    
# env = Map.loadFromFile("./maps/generated_map_N=50.E=125.D=20.txt")  
# env = Map.loadFromFile("./maps/generated_map_LOBSTER-N-10-E-12.0-D-20.txt")    
env = Map.loadFromFile("./maps/generated_map_LOBSTER-N-250-E-300.0-D-20.txt")    
# env = Map.loadFromFile("./maps/generated_map_N=70.E=245.D=20.txt")    

agent = MDPAgent(env, use_estimates=False, replan=True, observe_future=True)
evaluate(agent, 30, Nmean=5, printQuantile=0.9)
