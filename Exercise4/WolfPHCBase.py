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
import copy
from collections import defaultdict
		
class WolfPHCAgent(Agent):
	def __init__(self, learningRate, discountFactor, winDelta=0.01, loseDelta=0.1, initVals=0.0):
		super(WolfPHCAgent, self).__init__()
		self.discountFactor = discountFactor
		self.learningRate = learningRate
		self.actions = ['MOVE_UP', 'MOVE_DOWN', 'MOVE_LEFT', 'MOVE_RIGHT', 'KICK', 'NO_OP']
		self.numberOfActions = len(self.actions)

		self.winDelta = 0.0001
		self.loseDelta = 0.001
		
		# Initial empty lists 
		self.averagepolicy = {}
		self.C = {}
		self.policy = {}
		self.Q = {}

		# Extra Information required for learning 
		self.S_t_1 = self.S_t =None
		self.a_t_1 = None
		self.r_t = 0

		
		
	def setExperience(self, state, action, reward, status, nextState):
		if nextState != (-1,-1):
			if nextState not in self.C.keys():
				self.C[nextState]=0.0
				for Ai in self.actions:
				    self.Q[(nextState,Ai)]=0.0
				    self.policy[(nextState,Ai)]=1/len(self.actions) 
				    self.averagepolicy[(nextState,Ai)]=0 

		self.S_t_1 = state
		self.a_t_1 = action
		self.S_t = nextState
		self.r_t = reward

	def setState(self, state):	
		# Initialize the policy and the average policy 
		if state not in self.C.keys():
			self.C[state]=0.0
			for Ai in self.actions:
				self.Q[(state,Ai)]=0.0
				self.policy[(state,Ai)]=1/len(self.actions) 
				self.averagepolicy[(state,Ai)]=0 
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
		self.C[self.S_t_1] +=1	
		
		count = self.C[self.S_t_1]
		for i,act in enumerate(self.actions):
			self.averagepolicy[(self.S_t_1,act)] += (1/count)*(-self.averagepolicy[(self.S_t_1,act)] +self.policy[(self.S_t_1,act)])		
		calcuate = []	
		for act in self.actions:
			calcuate.append(self.averagepolicy[self.S_t_1,act] )

		return calcuate
	

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
		sum_policy , sum_avrg_policy = 0, 0 

		for i,action in enumerate(self.actions):
				sum_policy += self.policy[(self.S_t_1,action)]*self.Q[(self.S_t_1,action)]
				sum_avrg_policy += self.averagepolicy[(self.S_t_1,action)]*self.Q[(self.S_t_1,action)]
		if sum_policy>=sum_avrg_policy:
			lr = self.winDelta
		else:
			lr = self.loseDelta
			
		#Update probabilities of suboptimal actions
		p_moved = 0 
		
		for i,action in enumerate(self.actions):
			if action in suboptimal_sol.keys():
				p_moved += min(lr/len(suboptimal_sol),self.policy[(self.S_t_1,action)])		
				self.policy[(self.S_t_1,action)] -= min(lr/len(suboptimal_sol),self.policy[(self.S_t_1,action)])
				
		
		#Update probabilities of optimal actions
		for i,action in enumerate(self.actions):
			if action not in suboptimal_sol.keys():
				self.policy[(self.S_t_1,action)] += (p_moved)/(self.numberOfActions - len(suboptimal_sol))
		calcuate = []	
		for act in self.actions:
			calcuate.append(self.policy[self.S_t_1,act] )

		return calcuate
	
		

	def act(self):
		state_value_probabilities = []
		for Ai in self.actions:
		    state_value_probabilities.append(self.policy[(self.S_t,Ai)])
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
		if episodeNumber==10000:
			self.winDelta=0.001
			self.loseDelta=0.01
		if episodeNumber==20000:
			self.winDelta=0.01
			self.loseDelta=0.1
		if episodeNumber==30000:
			self.winDelta=0.05
			self.loseDelta=0.5
		if episodeNumber==40000:
			self.winDelta=0.1
			self.loseDelta=1
		self.learningRate = max((1-episodeNumber/50000),1e-5)*0.5	
		return self.loseDelta, self.winDelta, self.learningRate
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
	total_goals = 0 
	final_goals = 0 

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

			if reward[0] ==1:
				total_goals +=1
			print(episode,total_goals)
			if episode > 44999:
				if reward[0] == 1:
					final_goals +=1
				
			observation = nextObservation
	print(episode,total_goals)
	print(episode,final_goals)