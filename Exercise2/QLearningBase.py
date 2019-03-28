#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import itertools
import numpy as np
import random
from sys import maxsize


class QLearningAgent(Agent):
	
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(QLearningAgent, self).__init__()
		self.discountFactor = discountFactor
		self.learningRate = learningRate
		self.actions = ['DRIBBLE_UP', 'DRIBBLE_DOWN', 'DRIBBLE_LEFT', 'DRIBBLE_RIGHT', 'KICK']
		self.numberOfActions = 5
		self.epsilon = epsilon
		
		# --------------------- possible state-action --------------------------- #
		
		self.states = list(itertools.product(list(range(5)), list(range(6))))
		self.stateAction = list(itertools.product(self.states, self.actions))
		
		# --------------------- behavior, target policy and stateValue Q(s,a)  --------------------------- #
		
		self.behavior_policy = {(x, y): [0.2, 0.2, 0.2, 0.2, 0.2] for (x, y) in self.states}  # greedy policy
		# self.policy = {}
		self.stateValue = {}
		#  {((x, y), z): 0 for ((x, y), z) in self.stateAction}
		
		# --------------------- Resettable Variables  --------------------------- #
		
		self.stateS_t_1 = self.stateS_t =None
		self.actionS_t_1 = None
		self.rewardS_t = 0
		
	def setExperience(self, state, act, reward, status, nextState):
		if nextState != (-1,-1):
			for action in self.actions:
				if not (nextState,action) in self.stateValue:
					self.stateValue[(nextState,action)] = 0
				if not (state,action) in self.stateValue:
					self.stateValue[(state,action)] = 0
		self.stateS_t_1 = state
		self.actionS_t_1 = act
		self.stateS_t = nextState
		self.rewardS_t = reward
		
	def learn(self):
		
		# Update Q(s,a)
		prior = self.stateValue[(self.stateS_t_1, self.actionS_t_1)]
		max_value = (- maxsize)
		if self.stateS_t != (-1, -1):
			for action in self.actions:
				temp = self.stateValue[(self.stateS_t, action)]
				if temp > max_value:
					max_value = temp
			self.stateValue[(self.stateS_t_1, self.actionS_t_1)] += self.learningRate * (
					self.rewardS_t + self.discountFactor * max_value
					- self.stateValue[(self.stateS_t_1, self.actionS_t_1)]
			)
		else:
			self.stateValue[(self.stateS_t_1, self.actionS_t_1)] += self.learningRate * (
				self.rewardS_t - self.stateValue[(self.stateS_t_1, self.actionS_t_1)]
			)
			
		# Update behavior policy e-greedy μ
		# policy = []
		# for action in self.actions:
		# 	if self.stateValue[(self.stateS_t_1, action)] == max_value:
		# 		policy.append((1 - self.epsilon) / total_best_actions + self.epsilon / self.numberOfActions)
		# 	else:
		# 		policy.append(self.epsilon / self.numberOfActions)
	
		return self.stateValue[(self.stateS_t_1, self.actionS_t_1)] - prior
	
	
	
	
	def toStateRepresentation(self, state):
		if type(state) == str:
			return -1, -1
		else:
			return state[0]
	
	def setState(self, state):
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

		# state_value_probabilities = self.behavior_policy[self.stateS_t]
		# return np.random.choice(self.actions, p=state_value_probabilities)
		
	
	def setLearningRate(self, learningRate):
		self.learningRate = learningRate
	
	def setEpsilon(self, epsilon):
		self.epsilon = epsilon
		self.next_state = self.curr_state = None
		self.curr_action = None
	
	def reset(self):
		pass
	
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		return self.learningRate, self.epsilon


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=500)
	
	args = parser.parse_args()
	
	# Initialize connection with the HFO server
	hfoEnv = HFOAttackingPlayer(numOpponents=args.numOpponents, numTeammates=args.numTeammates, agentId=args.id)
	hfoEnv.connectToServer()
	
	# Initialize a Q-Learning Agent
	agent = QLearningAgent(learningRate=0.1, discountFactor=0.99, epsilon=0.1)
	numEpisodes = args.numEpisodes
	
	# Run training using Q-Learning
	numTakenActions = 0
	for episode in range(numEpisodes):
		status = 0
		observation = hfoEnv.reset()
		
		while status == 0:
			learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			agent.setLearningRate(learningRate)
			
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1
			
			nextObservation, reward, done, status = hfoEnv.step(action)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status,
			                    agent.toStateRepresentation(nextObservation))
			update = agent.learn()
			
			observation = nextObservation
