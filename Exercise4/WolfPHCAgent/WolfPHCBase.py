#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from sys import maxsize
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import numpy as np
		
class WolfPHCAgent(Agent):
	def __init__(self, learningRate, discountFactor, winDelta=0.01, loseDelta=0.1, initVals=0.0):
		super(WolfPHCAgent, self).__init__()
		self.discountFactor = discountFactor
		self.learningRate = learningRate
		self.actions = ['MOVE_UP', 'MOVE_DOWN', 'MOVE_LEFT', 'MOVE_RIGHT', 'KICK', 'NO_OP']
		self.numberOfActions = 6
		self.winDelta = winDelta
		self.loseDelta = loseDelta
		
		# Initial empty lists 
		self.tempPolicy = {}
		self.averagePolicy = {}
		self.averagePolicyCounts = {}
		self.policy = {}
		self.stateValue = {}	

		# Extra Information required for learning 
		self.stateS_t_1 = self.stateS_t =None
		self.actionS_t_1 = None
		self.rewardS_t = 0
	
		
		
	def setExperience(self, state, action, reward, status, nextState):
		if state != (-1,-1):

			# Initialize the policy for every undiscovered state policy
			if not nextState in self.policy:
				self.policy[nextState] = [1/6,1/6,1/6,1/6,1/6,1/6]

			# Initialize state-value function for every undiscovered state policy
			for action in self.actions:
				if not (nextState,action) in self.stateValue:
					self.stateValue[(nextState,action)] = 0
				if not (state,action) in self.stateValue:
					self.stateValue[(state,action)] = 0
		self.stateS_t_1 = state
		self.actionS_t_1 = action
		self.stateS_t = nextState
		self.rewardS_t = reward

	def learn(self):

		max_value = (- maxsize)
		total_best_actions = 0

		if self.stateS_t != (-1, -1):
			for action in self.actions:
				temp = self.stateValue[(self.stateS_t, action)]
				if temp > max_value:
					max_value = temp
					total_best_actions=1
				elif temp == max_value:
					total_best_actions += 1
			self.stateValue[(self.stateS_t_1, self.actionS_t_1)] += self.learningRate * (
					self.rewardS_t + self.discountFactor * max_value
					- self.stateValue[(self.stateS_t_1, self.actionS_t_1)]
			)
		else:
			self.stateValue[(self.stateS_t_1, self.actionS_t_1)] += self.learningRate * (
				self.rewardS_t - self.stateValue[(self.stateS_t_1, self.actionS_t_1)]
			)
			
		

	def act(self):
		state_value_probabilities = self.policy[self.stateS_t]
		return np.random.choice(self.actions, p=state_value_probabilities)
		

	def calculateAveragePolicyUpdate(self):
		self.averagePolicyCounts[self.stateS_t_1]+=1
		self.averagePolicy[self.stateS_t_1]+= (1/self.averagePolicyCounts[self.stateS_t_1])* \
			(self.policy[self.stateS_t_1] - self.averagePolicy[self.stateS_t_1] )
	

	def calculatePolicyUpdate(self):
		# Find suboptimal Solutions
		Q_max = -(maxsize)
		for action in self.actions:
				temp = self.stateValue[(self.stateS_t_1, action)]
				if temp > Q_max:
					Q_max = temp
		suboptimal_sol = {}

		for action in self.actions:
				temp = self.stateValue[(self.stateS_t_1, action)]
				if temp < Q_max:
					suboptimal_sol[action] = 1
		
		# Decide used learning rate 
		sum_policy = sum_avrg_policy = 0 
		for action in self.actions:
				sum_policy += self.policy[self.stateS_t_1].index(action)*self.stateValue[(self.stateS_t_1,action)]
				sum_avrg_policy += self.averagePolicy[self.stateS_t_1].index(action)*self.stateValue[(self.stateS_t_1,action)]
		lr = self.winDelta  if sum_policy>sum_avrg_policy else self.loseDelta
		p_moved = 0 
		for action in self.actions:
			if action in suboptimal_sol:
				p_moved += min(lr/len(suboptimal_sol),self.policy[self.stateS_t_1].index(action))
				index = self.policy[self.stateS_t_1].index(action)
				self.policy[self.stateS_t_1][index] -= min(lr/len(suboptimal_sol),self.policy[self.stateS_t_1].index(action))
			else:
				index = self.policy[self.stateS_t_1].index(action)
				self.policy[self.stateS_t_1][index] += p_moved/(self.numberOfActions - len(suboptimal_sol))

	
	def toStateRepresentation(self, state):
		if type(state) == str:
			return -1, -1
		else:
			attacker1 = tuple(state[0][0])
			attacker2 = tuple(state[0][1])
			ball = tuple(state[1][0])
			defender = tuple(state[2][0])
		return tuple((attacker1,attacker2,ball,defender))

	def setState(self, state):
		self.state = state
		if not state in self.averagePolicy:
			self.averagePolicyCounts[state] = 0
		# Initialize the policy and the average policy 
		if not state in self.policy:
				self.averagePolicy[state] =[1/6,1/6,1/6,1/6,1/6,1/6]
				self.policy[state] =[1/6,1/6,1/6,1/6,1/6,1/6]

	def setLearningRate(self,lr):
		self.learningRate = lr
		
	def setWinDelta(self, winDelta):
		self.winDelta = winDelta		
	def setLoseDelta(self, loseDelta):
		self.loseDelta = loseDelta
	
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		return self.loseDelta,self.winDelta,self.learningRate

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--numOpponents', type=int, default=1)
	parser.add_argument('--numAgents', type=int, default=2)
	parser.add_argument('--numEpisodes', type=int, default=100000)

	args=parser.parse_args()

	numOpponents = args.numOpponents
	numAgents = args.numAgents
	MARLEnv = DiscreteMARLEnvironment(numOpponents = numOpponents, numAgents = numAgents)

	agents = []
	for i in range(args.numAgents):
		agent = WolfPHCAgent(learningRate = 0.2, discountFactor = 0.99, winDelta=0.01, loseDelta=0.1)
		agents.append(agent)

	numEpisodes = args.numEpisodes
	numTakenActions = 0
	for episode in range(numEpisodes):	
		status = ["IN_GAME","IN_GAME","IN_GAME"]
		observation = MARLEnv.reset()
		
		while status[0]=="IN_GAME":
			for agent in agents:
				loseDelta, winDelta, learningRate = agent.computeHyperparameters(numTakenActions, episode)
				agent.setLoseDelta(loseDelta)
				agent.setWinDelta(winDelta)
				agent.setLearningRate(learningRate)
			actions = []
			perAgentObs = []
			agentIdx = 0
			for agent in agents:
				obsCopy = deepcopy(observation[agentIdx])
				perAgentObs.append(obsCopy)
				agent.setState(agent.toStateRepresentation(obsCopy))
				actions.append(agent.act())
				agentIdx += 1
			nextObservation, reward, done, status = MARLEnv.step(actions)
			numTakenActions += 1

			agentIdx = 0
			for agent in agents:
				agent.setExperience(agent.toStateRepresentation(perAgentObs[agentIdx]), actions[agentIdx], reward[agentIdx], 
					status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
				agent.learn()
				agent.calculateAveragePolicyUpdate()
				agent.calculatePolicyUpdate()
				agentIdx += 1
			
			observation = nextObservation
