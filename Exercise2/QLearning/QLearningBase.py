#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import itertools
import numpy as np
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
		
		self.states = list(itertools.product(list(range(6)), list(range(5))))
		self.stateAction = list(itertools.product(self.states, self.actions))
		
		# --------------------- behavior, target policy and stateValue Q(s,a)  --------------------------- #
		
		self.target_policy = {(x, y): [0.2, 0.2, 0.2, 0.2, 0.2] for (x, y) in self.states}  # greedy policy
		self.behavior_policy = {(x, y): np.random.choice(self.actions) for (x, y) in self.states}  # e-greedy policy
		self.stateValue = {((x, y), z): 0 for ((x, y), z) in self.stateAction}
		
		# --------------------- Resetable Variables  --------------------------- #
		self.next_state = self.curr_state = None
		self.curr_action = None
		self.curr_reward = 0
		
	def learn(self):
		
		# Update Q(s,a)
		max_value = (- maxsize)
		for action in self.actions:
			temp = self.stateValue[(self.next_state, action)]
			if temp > max_value:
				max_value = temp
		self.stateValue[(self.curr_state, self.curr_action)] += self.learningRate * (
				self.curr_reward + self.discountFactor * max_value
				- self.stateValue[(self.curr_state, self.curr_action)]
		)
		# Update behavior policy
		for action in self.actions:
			temp = self.stateValue[(self.stateS_t_2, action)]
			if temp > best_reward:
				best_reward = temp
				total_best_actions = 1
			elif temp == best_reward:
				total_best_actions += 1
		policy = []
		for action in self.actions:
			if self.stateValue[(self.curr_state, action)] == best_reward:
				policy.append((1 - self.epsilon) / total_best_actions + self.epsilon / self.numberOfActions)
			else:
				policy.append(self.epsilon / self.numberOfActions)
		
		return self.stateValue[(self.curr_state, self.curr_action)]
	
	def setExperience(self, state, action, reward, status, nextState):
		
		self.curr_state = state
		self.next_state = nextState
		self.curr_action = action
		self.curr_reward = reward
	
	def toStateRepresentation(self, state):
		return state[0]
	
	def setState(self, state):
		self.stateS_t = state
	
	def act(self):
		state_value_probabilities = self.policy[self.stateS_t]
		action = np.random.choice(self.actions, p=state_value_probabilities)
		return action
	
	def setLearningRate(self, learningRate):
		self.learningRate = learningRate
	
	def setEpsilon(self, epsilon):
		self.epsilon = epsilon
		self.next_state = self.curr_state = None
		self.curr_action = None
	
	def reset(self):
		self.curr_reward = 0
	
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
	agent = QLearningAgent(learningRate=0.1, discountFactor=0.99, epsilon=1.0)
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
