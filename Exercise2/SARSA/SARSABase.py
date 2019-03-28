#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import itertools
import numpy as np
from sys import maxsize
import random


class SARSAAgent(Agent):
	
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(SARSAAgent, self).__init__()
		self.discountFactor = discountFactor
		self.learningRate = learningRate
		self.actions = ['DRIBBLE_UP', 'DRIBBLE_DOWN', 'DRIBBLE_LEFT', 'DRIBBLE_RIGHT', 'KICK']
		self.numberOfActions = 5
		self.epsilon = epsilon
		
		# Initialize state-value function Q(s,a)
		self.states = list(itertools.product(list(range(5)), list(range(6))))
		self.stateAction = list(itertools.product(self.states, self.actions))
		
		# Bad initialization policy
		self.policy = {} 	 
		self.stateValue = {((x, y), z): 0 for ((x, y), z) in self.stateAction}
		
		self.stateS_t_2 = self.stateS_t_1 = self.stateS_t = None
		self.rewardS_t_1 = self.rewardS_t = None
		self.actionS_t_2 = self.actionS_t_1 = None
	
	def learn(self):
		if self.stateS_t != None:
			stateS_t_2 = (self.stateS_t_2, self.actionS_t_2)
			stateS_t_1 = (self.stateS_t_1, self.actionS_t_1)
			self.stateValue[stateS_t_2] += self.learningRate * (
					self.rewardS_t_1 + self.discountFactor * self.stateValue[stateS_t_1]
					- self.stateValue[stateS_t_2])
		else:
			stateS_t_2 = (self.stateS_t_2, self.actionS_t_2)
			self.stateValue[stateS_t_2] += self.learningRate * (self.rewardS_t_1 - self.stateValue[stateS_t_2])
		
		return self.stateValue[stateS_t_2]
	
	def setExperience(self, state, action, reward, status, nextState):
		
			self.stateS_t_2 = self.stateS_t_1
			self.stateS_t_1 = state
			self.stateS_t = nextState

			self.rewardS_t_1 = self.rewardS_t
			self.rewardS_t = reward
			
			self.actionS_t_2 = self.actionS_t_1
			self.actionS_t_1 = action
	
	def setState(self, state):		
		self.stateS_t = state
		
	

	def act(self):
		action_distribution={}
		for action in self.actions:
			action_distribution[action] = self.stateValue[(self.stateS_t,action)]
		maxValue = max(action_distribution.values())
		action_to_take = np.random.choice([k for k,v in action_distribution.items() if v == maxValue])

		if random.random() < self.epsilon:     
			action_to_take = self.possibleActions[random.randint(0,len(self.possibleActions)-1)]
		return action_to_take

	
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		return max(1 -episodeNumber/4250,1e-6), max(1 -episodeNumber/4500,1e-6) #325/500
	
	def toStateRepresentation(self, state):
		if type(state) == str:
			return -1, -1
		else:
			return state[0]
		
	def reset(self):
		self.reward = 0
		self.stateS_t_2 = self.stateS_t_1 = self.stateS_t = None
		self.rewardS_t_1 = self.rewardS_t = None
		self.actionS_t_2 = self.actionS_t_1 = None
	
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
	agent = SARSAAgent(learningRate=0.1, discountFactor=0.9, epsilon=0.2)
	total_goals = 0 
	final_goals = 0 
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
			# print(obsCopy, action, reward, nextObservation)
			
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
		if reward ==1:
			total_goals +=1
		print(episode,total_goals)
		if episode > 4499:
			if reward == 1:
				final_goals +=1
	print(episode,final_goals)
