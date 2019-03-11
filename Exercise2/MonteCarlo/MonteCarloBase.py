#!/usr/bin/env python3
# encoding utf-8
import sys
import itertools


from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse

class MonteCarloAgent(Agent):
	def __init__(self, discountFactor, epsilon, initVals=0.0):
		super(MonteCarloAgent, self).__init__()
		self.discountFactor = discountFactor
		self.actions = ['DRIBBLE_UP','DRIBBLE_DOWN','DRIBBLE_LEFT', 'DRIBBLE_RIGHT','KICK']
		self.epsilon = epsilon
		self.stateValue = {}
		states = list(itertools.product(list(range(6)), list(range(5))))
		stateAction = list(itertools.product(states, self.actions))
		self.stateValue = {((x, y), z) for ((x, y), z) in stateAction}
		
		

	def learn(self):
		raise NotImplementedError

	def toStateRepresentation(self, state):
		raise NotImplementedError

	def setExperience(self, state, action, reward, status, nextState):
		raise NotImplementedError

	def setState(self, state):
		raise NotImplementedError

	def reset(self):
		raise NotImplementedError

	def act(self):
		raise NotImplementedError

	def setEpsilon(self, epsilon):
		raise NotImplementedError

	def computeHyperparameters(self, numTakenActions, episodeNumber):
		raise NotImplementedError


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=500)

	args=parser.parse_args()

	#Init Connections to HFO Server
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a Monte-Carlo Agent
	agent = MonteCarloAgent(discountFactor = 0.99, epsilon = 1.0)
	numEpisodes = args.numEpisodes
	numTakenActions = 0
	# Run training Monte Carlo Method
	for episode in range(numEpisodes):	
		agent.reset()
		observation = hfoEnv.reset()
		status = 0

		while status==0:
			epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1
			nextObservation, reward, done, status = hfoEnv.step(action)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			observation = nextObservation

		agent.learn()
