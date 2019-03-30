#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
import itertools
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
import numpy as np
from sys import maxsize
from copy import deepcopy
import argparse
		
class IndependentQLearningAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(IndependentQLearningAgent, self).__init__()
		self.discountFactor = discountFactor
		self.learningRate = learningRate
		self.actions = ['MOVE_UP', 'MOVE_DOWN', 'MOVE_LEFT', 'MOVE_RIGHT', 'KICK', 'NO_OP']
		self.numberOfActions = 6
		self.epsilon = epsilon
		
		# --------------------- possible state-action --------------------------- #
		
		self.states = list(itertools.product(list(range(5)), list(range(6))))
		self.stateAction = list(itertools.product(self.states, self.actions))
		
		# --------------------- behavior, target policy and stateValue Q(s,a)  --------------------------- #
		
		self.stateValue = {}
		
		# --------------------- Resettable Variables  --------------------------- #
		
		self.stateS_t_1 = self.stateS_t =None
		self.actionS_t_1 = None
		self.rewardS_t = 0

	def setExperience(self, state, act, reward, status, nextState):
		# if nextState != (-1,-1):
		for action in self.actions:
			if (nextState,action) not in self.stateValue.keys():
				self.stateValue[(nextState,action)] = 0
		self.stateS_t_1 = state
		self.actionS_t_1 = act
		self.stateS_t = nextState
		self.rewardS_t = reward
		
	def learn(self):
		
		# Update Q(s,a)
		prior = self.stateValue[(self.stateS_t_1, self.actionS_t_1)]
		max_value = (-maxsize)
		
		# if self.stateS_t != (-1, -1):
		for action in self.actions:
			temp = self.stateValue[(self.stateS_t, action)]
			if temp > max_value:
				max_value = temp
		self.stateValue[(self.stateS_t_1, self.actionS_t_1)] += self.learningRate * (
				self.rewardS_t + self.discountFactor * max_value - prior
		)
		# else:
		# 	self.stateValue[(self.stateS_t_1, self.actionS_t_1)] += self.learningRate * (
		# 		self.rewardS_t - prior
		# 	)
		return self.stateValue[(self.stateS_t_1, self.actionS_t_1)] -prior
	
	def toStateRepresentation(self, state):
		if type(state) == str:
			return -1, -1
		else:
			attacker1 = tuple(state[0][0])
			attacker2 = tuple(state[0][1])
			ball = tuple(state[1][0])
			defender = tuple(state[2][0])
		return (attacker1,attacker2,ball,defender)

	def setState(self, state):
		for action in self.actions:
			if  (state,action) not in self.stateValue.keys():
					self.stateValue[(state,action)] = 0
		self.stateS_t = state
	
	def act(self):
		action_distribution = dict()
		for action in self.actions:
			action_distribution[action] =  self.stateValue[(self.stateS_t,action)]
		maxValue = max(action_distribution.values())
		action_to_take = np.random.choice([k for k,v in action_distribution.items() if v == maxValue])

		if random.random() < self.epsilon:      # e-greedy action
			action_to_take = self.actions[random.randint(0,len(self.actions)-1)]
		return action_to_take


	
	def setLearningRate(self, learningRate):
		self.learningRate = learningRate
	
	def setEpsilon(self, epsilon):
		self.epsilon = epsilon
	
	def reset(self):
		self.next_state = self.curr_state = None
		self.curr_action = None

	def computeHyperparameters(self, numTakenActions, episodeNumber):
		return max(1 - episodeNumber/45000,1e-5), max(1 - episodeNumber/45000,1e-6)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--numOpponents', type=int, default=1)
	parser.add_argument('--numAgents', type=int, default=2)
	parser.add_argument('--numEpisodes', type=int, default=50000)

	args=parser.parse_args()

	MARLEnv = DiscreteMARLEnvironment(numOpponents = args.numOpponents, numAgents = args.numAgents)
	agents = []
	for i in range(args.numAgents):
		agent = IndependentQLearningAgent(learningRate = 0.1, discountFactor = 0.9, epsilon = 1)
		agents.append(agent)
	total_goals = 0 
	final_goals = 0 
	numEpisodes = args.numEpisodes
	numTakenActions = 0
	for episode in range(numEpisodes):	
		status = ["IN_GAME","IN_GAME","IN_GAME"]
		observation = MARLEnv.reset()
		totalReward = 0.0
		timeSteps = 0
			
		while status[0]=="IN_GAME":
			for agent in agents:
				learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
				agent.setEpsilon(epsilon)
				agent.setLearningRate(learningRate)
			actions = []
			stateCopies = []
			for agentIdx in range(args.numAgents):
				obsCopy = deepcopy(observation[agentIdx])
				stateCopies.append(obsCopy)
				agents[agentIdx].setState(agent.toStateRepresentation(obsCopy))
				actions.append(agents[agentIdx].act())
			numTakenActions += 1
			nextObservation, reward, done, status = MARLEnv.step(actions)
			
			for agentIdx in range(args.numAgents):
				agents[agentIdx].setExperience(agent.toStateRepresentation(stateCopies[agentIdx]), actions[agentIdx], reward[agentIdx], 
					status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
				agents[agentIdx].learn()
			if reward[0] == 1 :
				total_goals+=1
			print(episode,total_goals)
			if episode > 44999:
				if reward[0] == 1:
					final_goals +=1
				
			observation = nextObservation
	print(episode,total_goals)
	print(episode,final_goals)
				
