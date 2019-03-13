#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from sys import maxsize
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import numpy as np
from operator import add,sub
import numpy as np
import copy
from collections import defaultdict
		
class WolfPHCAgent(Agent):
	def __init__(self, learningRate, discountFactor, winDelta=0.01, loseDelta=0.1, initVals=0.0):
		super(WolfPHCAgent, self).__init__()
		self.discountFactor = discountFactor
		self.learningRate = learningRate
		self.actions = ['MOVE_UP', 'MOVE_DOWN', 'MOVE_LEFT', 'MOVE_RIGHT', 'KICK', 'NO_OP']
		self.numberOfActions = len(self.actions)

		self.winDelta = winDelta
		self.loseDelta = loseDelta
		
		# Initial empty lists 
		self.averagePolicy = {}
		self.averagePolicyCounts = {}
		self.policy = {}
		self.Q = {}

		# Extra Information required for learning 
		self.S_t_1 = self.S_t =None
		self.a_t_1 = None
		self.r_t = 0

		
		
	def setExperience(self, state, action, reward, status, nextState):
		if state != (-1,-1):

			# Initialize the policy for every undiscovered state 
			if nextState not in self.policy:
				self.policy[nextState] = [1/self.numberOfActions]*self.numberOfActions
				self.averagePolicy[nextState] = [0]*self.numberOfActions

			# Initialize state-action function for every undiscovered state
			for action in self.actions:
				if (nextState,action) not in self.Q:
					self.Q[(nextState,action)] = 0
			
		self.S_t_1 = state
		self.a_t_1 = action
		self.S_t = nextState
		self.r_t = reward

	def setState(self, state):
		
		# Initialize the policy and the average policy 
		
		if state not in self.policy.keys():
				self.averagePolicy[state] =  [0]*self.numberOfActions
				self.policy[state] =  [1/self.numberOfActions]*self.numberOfActions
		for action in self.actions:
			if (state,action) not in self.Q:
				self.Q[(state,action)] = 0
		self.S_t = state 
	
	def learn(self):
		
		prev_Q = copy.deepcopy(self.Q[(self.S_t_1, self.a_t_1)])
		max_value = (- maxsize)		
		if self.S_t != (-1, -1):
			for action in self.actions:
				temp = self.Q[(self.S_t, action)]
				if temp > max_value:
					max_value = temp
			self.Q[(self.S_t_1, self.a_t_1)] += self.learningRate * (
					self.r_t +  self.discountFactor*max_value
					- self.Q[(self.S_t_1, self.a_t_1)]
			)
		else:
			self.Q[(self.S_t_1, self.a_t_1)] += self.learningRate * (
				self.r_t - self.Q[(self.S_t_1, self.a_t_1)]
			)			
		return self.Q[(self.S_t_1, self.a_t_1)] - prev_Q

	def calculateAveragePolicyUpdate(self):	
		if self.S_t_1 not in self.averagePolicyCounts:
			self.averagePolicyCounts[self.S_t_1] = 1
		else:
			self.averagePolicyCounts[self.S_t_1] +=1	
		
		count = self.averagePolicyCounts[self.S_t_1]
		list_trial = []
		for i,_ in enumerate(self.actions):
			list_trial.append(self.averagePolicy[self.S_t_1][i] + (1/count)*(self.averagePolicy[self.S_t_1][i] -self.policy[self.S_t_1][i]))		
		self.averagePolicy[self.S_t_1] =  list_trial 
		# return self.averagePolicy[self.S_t_1] - hard_copy
	
	def reset(self):
		pass

	

	def calculatePolicyUpdate(self):

		# Find suboptimal Solutions
		Q_max = -(maxsize)
		for action in self.actions:
				temp = self.Q[(self.S_t_1, action)]
				if temp >= Q_max:
					Q_max = temp

		suboptimal_sol = {}

		for action in self.actions:
				temp = self.Q[(self.S_t_1, action)]
				if temp != Q_max:
					suboptimal_sol[action] = 1
					
		# Decide used learning rate 
		sum_policy = sum_avrg_policy = 0 

		for i,action in enumerate(self.actions):
				sum_policy += self.policy[self.S_t_1][i]*self.Q[(self.S_t_1,action)]
				sum_avrg_policy += self.averagePolicy[self.S_t_1][i]*self.Q[(self.S_t_1,action)]
		if sum_policy>=sum_avrg_policy:
			lr = self.winDelta
		else:
			lr = self.loseDelta
			
		#Update probabilities of suboptimal actions
		p_moved = 0 
		
		for i,action in enumerate(self.actions):
			if action in suboptimal_sol:
				p_moved += min(lr/len(suboptimal_sol),self.policy[self.S_t_1][i])		
				self.policy[self.S_t_1][i] -= min(lr/len(suboptimal_sol),self.policy[self.S_t_1][i])
				
		
		#Update probabilities of optimal actions
		for i,action in enumerate(self.actions):
			if action not in suboptimal_sol:
				self.policy[self.S_t_1][i] += (p_moved)/(self.numberOfActions - len(suboptimal_sol))
		return self.policy[self.S_t_1]
		

	def act(self):
		state_value_probabilities = self.policy[self.S_t]
		action = np.random.choice(self.actions, p=state_value_probabilities)
		return action
		
		
	def toStateRepresentation(self, state):
		if type(state) == str:
			return -1, -1
		else:
			attacker1 = tuple(state[0][0])
			attacker2 = tuple(state[0][1])
			ball = tuple(state[1][0])
			defender = tuple(state[2][0])
		return tuple((attacker1,attacker2,ball,defender))
	

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
	parser.add_argument('--numEpisodes', type=int, default=50000)

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
	goals = 0 
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
			if reward[0] == 1:
				goals +=1
			if done and reward[0] ==1 :
				print("Goals:",goals," at episode:",episode)
			
			
			
			observation = nextObservation
	print("Total goals:",goals)
