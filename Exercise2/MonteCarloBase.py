#!/usr/bin/env python3
# encoding utf-8

import itertools
import numpy as np
from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
from sys import maxsize

class MonteCarloAgent(Agent):
	
	def __init__(self, discountFactor, epsilon, initVals=0.0):
		super(MonteCarloAgent, self).__init__()
		self.discountFactor = discountFactor
		self.actions = ['DRIBBLE_UP', 'DRIBBLE_DOWN', 'DRIBBLE_LEFT', 'DRIBBLE_RIGHT', 'KICK']
		self.numberOfActions = 5
		self.epsilon = epsilon
		self.stateValue = {}
		
		# Initialize state-value function Q(s,a)
		self.states = list(itertools.product(list(range(5)), list(range(6))))
		self.stateAction = list(itertools.product(self.states, self.actions))
		
		# Bad initialization policy
		self.stateValue = {((x, y), z): 0 for ((x, y), z) in self.stateAction}
		self.visited = {}
		self.visited_on_this_episode = {}
		
		# TODO: Not a soft-policy, make it stochastic
		
		self.policy = {(x, y): [0.2, 0.2, 0.2, 0.2, 0.2] for (x, y) in self.states}
		self.reward = {}
		self.visited_at = {}
		self.counter = 0
	
	def toStateRepresentation(self, state):
		return state[0]
	
	def setState(self, state):
		
		self.currentState = state
	
	def print_Results(self):
		print(self.policy)
		print(self.stateValue)
	
		
	def learn(self):
		
		# TODO: Is it efficiently computed ? Can I compute the G_t more efficiently for all states ?
		
		# Q(s,a) evaluation, MC - control
		# print(self.reward)
		# for stateValue, timestep in self.visited_on_this_episode.items():
			# print(stateValue)
			# Compute expected return
			# G_t = 0
		
			# for i in range(timestep,self.counter-1):
			# 	gamma = self.discountFactor ** (i-timestep)
			# 	G_t += (gamma)*self.reward[i+1]
			#
			# self.stateValue[stateValue] += (1 / self.visited[stateValue])*(G_t - self.stateValue[stateValue])
		
		G_t = 0
		for i in reversed(range(self.counter-2)):
			G_t = self.reward[i] + G_t*self.discountFactor
			self.stateValue[self.visited_at[i]] += (1 / self.visited[self.visited_at[i]]) * (G_t - self.stateValue[self.visited_at[i]])
			
		
		# e-greedy policy π
		for (state, _), _ in self.visited_on_this_episode.items():
			best_reward = (-maxsize)
			total_best_actions = 0
			for action in self.actions:
				temp = self.stateValue[(state, action)]
				if temp > best_reward:
					best_reward = temp
					total_best_actions = 1
				elif temp == best_reward:
					total_best_actions += 1
			policy = []
			for action in self.actions:
				if self.stateValue[(state, action)] == best_reward:
					policy.append((1 - self.epsilon)/total_best_actions + self.epsilon / self.numberOfActions)
				else:
					policy.append(self.epsilon/self.numberOfActions)
			self.policy[state] = policy
	
	
	def setExperience(self, state, action, reward, status, nextState):
		if not (state, action) in self.visited_on_this_episode:
			# self.visited_at[self.counter] = (state, action)
			self.visited_on_this_episode[(state, action)] = 1
			if (state, action) in self.visited:
				self.visited[(state, action)] += 1
			else:
				self.visited[(state, action)] = 1
		self.reward[self.counter] = reward
		self.visited_at[self.counter] = (state, action)
		self.counter += 1
	
	
	def reset(self):
		self.reward = {}
		self.visited_at = {}
		self.counter = 0
		self.visited_on_this_episode = {}
	
	def act(self):
		probability_scores = self.policy[self.currentState]
		return np.random.choice(self.actions, p=probability_scores)
		
	def setEpsilon(self, epsilon):
		self.epsilon = epsilon
	
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		
		epsilon = self.epsilon * 0.99
		if episodeNumber == 1500:
			epsilon = self.epsilon /1000
		return  epsilon
		


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser() 
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=500)
	parser.add_argument('--epsilon', type=float, default=1)
	args = parser.parse_args()

	

	
	# Init Connections to HFO Server
	hfoEnv = HFOAttackingPlayer(numOpponents=args.numOpponents, numTeammates=args.numTeammates, agentId=args.id)
	hfoEnv.connectToServer()
	
	# Initialize a Monte-Carlo Agent
	agent = MonteCarloAgent(discountFactor=0.9, epsilon=args.epsilon)
	numEpisodes = args.numEpisodes
	numTakenActions = 0
	
	# Run training Monte Carlo Method
	for episode in range(numEpisodes):
		agent.reset()
		
		observation = hfoEnv.reset()
		status = 0
		
		while status == 0:
			epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1
			nextObservation, reward, done, status = hfoEnv.step(action)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status,
			                    agent.toStateRepresentation(nextObservation))
			observation = nextObservation
		agent.learn()
	# agent.print_Results()
