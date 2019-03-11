#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import itertools
import numpy as np


class SARSAAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(SARSAAgent, self).__init__()
		self.discountFactor = discountFactor
		self.learningRate = learningRate
		self.actions = ['DRIBBLE_UP', 'DRIBBLE_DOWN', 'DRIBBLE_LEFT', 'DRIBBLE_RIGHT', 'KICK']
		self.numberOfActions = 5
		self.epsilon = epsilon
		
		# Initialize state-value function Q(s,a)
		self.states = list(itertools.product(list(range(7)), list(range(6))))
		self.stateAction = list(itertools.product(self.states, self.actions))
		
		# Bad initialization policy
		self.stateValue = {((x, y), z): 0 for ((x, y), z) in self.stateAction}
		
		self.stateS_t_2 = self.stateS_t_1 = self.stateS_t = None
		self.rewardS_t_1 = self.rewardS_t = None
		self.actionS_t_2 = self.actionS_t_1 = None
	
	def learn(self):
		stateS_t_2 = (self.stateS_t_2,self.rewardS_t_2)
		stateS_t_1 = (self.stateS_t_1,self.rewardS_t_1)
		self.stateValue[stateS_t_1] += self.learningRate * (
				self.prev_reward + self.discountFactor * self.stateValue[stateS_t_2]
				- self.self.stateValue[stateS_t_1])
		return self.stateValue[stateS_t_1]
	
	def setExperience(self, state, action, reward, status, nextState):
		
		self.stateS_t_2 = self.stateS_t_1
		self.stateS_t_1 = state
		self.stateS_t = nextState
		
		self.rewardS_t_2 = self.rewardS_t_1
		self.rewardS_t_1 = reward
		
		self.actionS_t_1 = self.actionS_t
		self.actionS_t = reward
	
	def setState(self, state):
		self.stateS_t = state[0]

	def act(self):
		state_value_probabilities = self.policy[self.currentState]
		return np.random.choice(self.actions, p=state_value_probabilities)
	
	
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		# TODO: Compute Learning a and exploration factor e
		if episodeNumber % 100 == 0 :
			epsilon, learning_rate = self.epsilon / 10, self.learningRate / 10
		else:
			epsilon, learning_rate  = self.epsilon, self.learningRate
		return epsilon,learning_rate
	
	def toStateRepresentation(self, state):
		return state[0]
	
	def reset(self):
		self.reward = 0
	
	def setLearningRate(self, learningRate):
		self.learningRate = learningRate
	
	def setEpsilon(self, epsilon):
		self.epsilon = epsilon


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=500)
	
	args = parser.parse_args()
	
	numEpisodes = args.numEpisodes
	# Initialize connection to the HFO environment using HFOAttackingPlayer
	hfoEnv = HFOAttackingPlayer(numOpponents=args.numOpponents, numTeammates=args.numTeammates, agentId=args.id)
	hfoEnv.connectToServer()
	
	# Initialize a SARSA Agent
	agent = SARSAAgent(learningRate=0.1, discountFactor=0.9, epsilon=0.5)
	
	# Run training using SARSA
	numTakenActions = 0
	for episode in range(numEpisodes):
		agent.reset()
		status = 0
		
		observation = hfoEnv.reset()
		nextObservation = None
		epsStart = True
		
		while status == 0:
			learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			agent.setLearningRate(learningRate)
			
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1
			
			nextObservation, reward, done, status = hfoEnv.step(action)
			print(obsCopy, action, reward, nextObservation)
			
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status,
			                    agent.toStateRepresentation(nextObservation))
			
			if not epsStart:
				agent.learn()
			else:
				epsStart = False
			
			observation = nextObservation
		
		# Update the last encountered state of the episode
		agent.setExperience(agent.toStateRepresentation(nextObservation), None, None, None, None)
		agent.learn()
