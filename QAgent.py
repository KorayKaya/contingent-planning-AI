from environment import Map
from agent import QAgent, QAgentConfig
from utilities import evaluate

scores = []

env = Map.loadFromFile("./maps/generated_map_N=70.E=245.D=20.txt")
config = QAgentConfig(
    initialQ=5000, 
    learning_rate=1.0,
    variance_weight=0, 
    discount=0.999,
    epsilon=0.0
    )
agent = QAgent(env, config)

evaluate(agent, 2000, Nmean=100, printQuantile=0.9)