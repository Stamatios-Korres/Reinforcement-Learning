#!/usr/bin/env python3
# encoding utf-8
from sys import path
from os.path import dirname as dir
path.append(dir(path[0]))

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import argparse
from collections import defaultdict
import numpy as np

class IndependentQLearningAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(IndependentQLearningAgent, self).__init__()
		self.learningRate=learningRate
		self.discountFactor = discountFactor
		self.initVals=initVals
		self.epsilon = epsilon
		self.goals=0
		self.rewards=0
		self.outofbounds=0
		self.timeouts=0
		self.verbose=False
		self.q = dict()


	def _add_state_to_q(self, state):
		if not state in self.q.keys():
			self.q[state]=dict()
			for a in self.possibleActions:
				self.q[state][a] = 0

	def setExperience(self, state, action, reward, status, nextState):
		if self.verbose:
			print('Setting experience State: %s\tAction: %s\tReward: %s\tStatus: %s\tNext State: %s' % (state, action, reward, status, nextState))
		self.experience = {'state': state, 'action':action, 'reward': reward, 'nextState': nextState, 'status': status}
		self._add_state_to_q(state)
		self._add_state_to_q(nextState)

		##### REMOVE LATER
		if status=='GOAL':
			self.goals+=1
			if reward==1:
				self.rewards+=1
		elif status=='OUT_OF_BOUNDS':
			self.outofbounds+=1
		elif status=='TIMEOUT':
			self.timeouts+=1


	

	def learn(self):
		pass
		s = self.currentState
		if s != self.experience['state']:
			raise Exception('State not equals to state')
		a = self.experience['action']
		qs = self.q[s][a]
		if self.experience['status']=='IN_GAME':
			s_n = self.experience['nextState']
			max_qs_n = max(self.q[s_n].values())
		else:
			s_n = 'TERMINAL STATE'
			max_qs_n = 0
		r = self.experience['reward']
		dq = self.learningRate * (r + self.discountFactor * max_qs_n - qs)
		self.q[s][a] += dq
		if self.verbose:
			print('Learning - State: %s Action: %s Reward: %.3f Next State: %s Max Q(s, a) %.3f' % (s,a,r,s_n,max_qs_n))
		return dq


	def act(self):
		qvalues = self.q[self.currentState]
		maxValue = max(qvalues.values())
		action = np.random.choice([k for k,v in qvalues.items() if v == maxValue])

		if random.random() < self.epsilon:      # e-greedy action
			action = self.possibleActions[random.randint(0,len(self.possibleActions)-1)]

		# policy=dict()
		# for i, a in enumerate(self.possibleActions):
		# 	p = float(self.epsilon)/float(len(self.possibleActions))
		# 	if a == maxQvalueAction:
		# 		p += 1-self.epsilon
		# 	policy[a] = p
		# action = np.random.choice(list(policy.keys()), p=list(policy.values()))
		# if self.verbose:
		# 	print(20*'\\/' + 'New Action' + 20*'\\/')
		# 	print('Current state: ', self.currentState)
		# 	print('Q values: ', qvalues)
		# 	print("Policy: ", policy)
		# 	print('Action: ', action)
		return action

	def toStateRepresentation(self, state):
		if isinstance(state, list):
			out = (tuple(state[0][0]), tuple(state[0][1]), tuple(state[1][0]), tuple(state[2][0]))
		elif isinstance(state, str):
			out = state
		else:
			raise Warning('State not a list or a string?')
		return  out

	def setState(self, state):
		self._add_state_to_q(state)
		self.currentState=state

	def setLearningRate(self, learningRate):
		self.learningRate = learningRate

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon
		
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		learningRate = 0.1
		# epsilon = 0.5 *  (((5000-episodeNumber)/5000) ** 2)
		# epsilon = 1.0 - min(1, episode/50000.0)
		epsilon = max((0.97 ** (episodeNumber/500)),0.1)
		return learningRate, epsilon

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--numOpponents', type=int, default=1)
	parser.add_argument('--numAgents', type=int, default=2)
	parser.add_argument('--numEpisodes', type=int, default=50000)

	args=parser.parse_args()

	MARLEnv = DiscreteMARLEnvironment(numOpponents = args.numOpponents, numAgents = args.numAgents)
	agents = []
	for i in range(args.numAgents):
		agent = IndependentQLearningAgent(learningRate = 0.1, discountFactor = 0.9, epsilon = 1.0)
		agents.append(agent)

	numEpisodes = args.numEpisodes
	numTakenActions = 0
	episodeOutcomes = []
	f = open('progress.out','w')
	for episode in range(numEpisodes+1):	
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
			if done[0]:
				print(nextObservation, reward, done, status)
			for agentIdx in range(args.numAgents):
				agents[agentIdx].setExperience(agent.toStateRepresentation(stateCopies[agentIdx]), actions[agentIdx], reward[agentIdx], 
												status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
				agents[agentIdx].learn()
				
			observation = nextObservation

			# if reward[0]==1:
			# 	episodeOutcomes.append(1)
			# else:
			# 	episodeOutcomes.append(0)

		i=0
		if episode % 500 == 0:
			f.write('Episodes: %s\tGoals: %d\tOut of bounds: %s\tTotal reward %.0f\n' \
			        %(episode, agents[i].goals, agents[i].outofbounds, agents[i].rewards))
			f.flush()
	f.close()
