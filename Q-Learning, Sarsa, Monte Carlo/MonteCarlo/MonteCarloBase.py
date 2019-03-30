#!/usr/bin/env python3
# encoding utf-8

import itertools
import numpy as np
from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
from sys import maxsize
import random 
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
		self.list_visited_on_this_episode = []
		self.timefound = {}

		
		# TODO: Not a soft-policy, make it stochastic
	
		self.reward = {}
		self.counter = 0
	
	def toStateRepresentation(self, state):
		return state[0]
		
	
	def setState(self, state):		
		self.currentState = state
			
	def learn(self):
		return_list = [] 		
		for (state,action) in self.list_visited_on_this_episode:
			G_t = 0 
			start = self.timefound[(state,action)]
			for i in range(start,self.counter):
				gamma = self.discountFactor ** (i-start)
				G_t += (gamma)*self.reward[i+1]
			self.stateValue[(state,action)] += (1 / self.visited[(state,action)])*(G_t - self.stateValue[(state,action)])
			return_list.append(self.stateValue[(state,action)])
		return 'ggg',return_list
	
	
	def setExperience(self, state, action, rew, status, nextState):
		if (state, action) not in self.list_visited_on_this_episode:
			self.list_visited_on_this_episode.append((state, action))
			self.timefound[(state, action)]= self.counter
			if (state, action) in self.visited.keys():
				self.visited[(state, action)] += 1
			else:
				self.visited[(state, action)] = 1
		self.counter += 1
		self.reward[self.counter] = rew
	
	def reset(self):
		self.reward = {}
		self.timefound = {}
		self.counter = 0
		self.list_visited_on_this_episode =  []
	
	def act(self):
		action_distribution={}
		for action in self.actions:
			action_distribution[action] = self.stateValue[(self.currentState,action)]
		maxValue = max(action_distribution.values())
		action_to_take = np.random.choice([k for k,v in action_distribution.items() if v == maxValue])

		if random.random() < self.epsilon:     
			action_to_take = self.actions[random.randint(0,len(self.actions)-1)]
		return action_to_take
		# probability_scores = self.policy[self.currentState]
		# return np.random.choice(self.actions, p=probability_scores)
		
	def setEpsilon(self, epsilon):
		self.epsilon = epsilon
	
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		# epsilon = max(1 -episodeNumber/4900,0) -> 100/500
		# epsilon = max(1 -episodeNumber/5500,1e-4)
		
		epsilon = max(1 -episodeNumber/4500,1e-6) # 2091 total 370/500 
		# epsilon = 1 -episodeNumber/5000
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
	agent = MonteCarloAgent(discountFactor=0.99, epsilon=args.epsilon)
	numEpisodes = args.numEpisodes
	numTakenActions = 0
	total_goals = 0 
	final_goals = 0 
	
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
			if reward ==1:
				total_goals +=1
				print(total_goals,episode)
			if episode > 4499:
				if reward == 1:
					final_goals +=1
			
		agent.learn()
	# agent.print_Results()
	print("Total goals are:", total_goals,final_goals)
