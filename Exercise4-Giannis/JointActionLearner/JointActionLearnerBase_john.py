#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import numpy as np
		
class WolfPHCAgent(Agent):
	def __init__(self, learningRate, discountFactor, winDelta, loseDelta):
		super(WolfPHCAgent, self).__init__()
		self.winDelta = winDelta
		self.loseDelta = loseDelta
		self.alpha = learningRate
		self.gamma = discountFactor
		self.Actions = self.possibleActions
		self.Q = {}
		self.policy = {}
		self.averagepolicy = {}
		self.C = {}
		self.current_state = None
		self.status = 'IN_GAME'
		self.st_act_re_next = []

	#Used to sample from policy using the probability of each action
	def weighted_choice(self,state):
		choices , probs = [],[]
		if state not in self.C.keys():
			i = np.random.randint(len(self.Actions))
			return self.Actions[i]
		for act in self.Actions:
			choices.append(act)
			probs.append(self.policy[(state,act)])
		return np.random.choice(choices, 1, p=probs)[0]

	def setExperience(self, state, action, reward, status, nextState):
		if state not in self.C.keys():
			self.C[state]=0.0
			for Ai in self.Actions:
				self.Q[(state,Ai)]=0.0
				self.policy[(state,Ai)]=1/len(self.Actions) 
				self.averagepolicy[(state,Ai)]=0 
		self.st_act_re_next = [(state,action),(reward,nextState)]
		self.status = status

	def learn(self):
		action_distribution = {}
		current_state_action = self.st_act_re_next[0]
		reward, next_state = self.st_act_re_next[1][0], self.st_act_re_next[1][1]
		prior_value = deepcopy(self.Q[current_state_action])
		if (self.status == "IN_GAME"):
			for action in self.Actions:
				if ((next_state,action)) in self.Q.keys():
					action_distribution[action] = self.Q[(next_state,action)] 
				else:
					action_distribution[action]=0.0
			best_action = max(action_distribution.keys(), key=(lambda key: action_distribution[key]))
			td_target = reward + self.gamma * action_distribution[best_action]
		else:
			td_target = reward
		td_delta = td_target - self.Q[current_state_action]
		self.Q[current_state_action] += self.alpha * td_delta
		current_value = deepcopy(self.Q[current_state_action])
		return current_value - prior_value


	def act(self):
		action_to_take = self.weighted_choice(self.current_state)
		return action_to_take

	def calculateAveragePolicyUpdate(self):
		state = self.current_state
		expected_output =[]
		self.C[state]+=1
		#for every action in the current state update average policy
		for act in self.Actions:
			self.averagepolicy[(state,act)]+=(self.policy[(state,act)]-self.averagepolicy[(state,act)])/self.C[state]
			expected_output.append(self.averagepolicy[(state,act)])
		return expected_output

	def suboptimal_actions(self):
		#Calculate maximum value for Q
		Qmax,state = [], self.current_state
		for act in self.Actions:
			if (state,act) in self.Q.keys():
				Qmax.append(self.Q[(state,act)])
			else: Qmax.append(0.0)
		Aopt , Asubopt = [], []
		#If state not yet seen, consider all actions optimal
		if state not in self.C.keys():
			return Asubopt, self.Actions
		for act in self.Actions:
			if self.Q[(state,act)]==max(Qmax):
				Aopt.append(act)
			else: Asubopt.append(act)	
		return Asubopt, Aopt

	def choose_learning_rate(self):
		state = self.current_state
		policySum, averagepolicySum =0, 0
		for act in self.Actions:
			policySum+=self.policy[(state,act)]*self.Q[(state,act)]
			averagepolicySum+=self.averagepolicy[(state,act)]*self.Q[(state,act)]
		if policySum>=averagepolicySum: return self.winDelta
		else: return self.loseDelta

	def calculatePolicyUpdate(self):
		Suboptimal, Optimal = self.suboptimal_actions()
		delta = self.choose_learning_rate()
		p_moved, state = 0, self.current_state
		for act in Suboptimal:
			value = min((delta/len(Suboptimal)),self.policy[(state,act)])	
			p_moved+=value
			self.policy[(state,act)]-= value	
		for act in Optimal:
			self.policy[(state,act)]+= p_moved/(len(self.Actions)-len(Suboptimal))	
		expected_output =[]
		for act in self.Actions:
			expected_output.append(self.policy[(state,act)])
		return expected_output

	def toStateRepresentation(self, State):
		if type(State)== str:
			return State
		attacker1 = (State[0][0][0],State[0][0][1])
		attacker2 = (State[0][1][0],State[0][1][1])
		ball = (State[2][0][0],State[2][0][1])
		defender = (State[1][0][0],State[1][0][1])
		return ((attacker1,attacker2),ball,defender)
	
	def setLearningRate(self,lr):
		self.alpha = lr

	def setState(self, state):
		self.current_state = state

	def setWinDelta(self, winDelta):
		self.winDelta = winDelta

	def setLoseDelta(self, loseDelta):
		self.loseDelta = loseDelta

	def computeHyperparameters(self, numTakenActions, episodeNumber):
		if episodeNumber==10000:
			self.winDelta=0.01
			self.loseDelta=0.1
		if episodeNumber==20000:
			self.winDelta=0.1
			self.loseDelta=1
		if episodeNumber==30000:
			self.winDelta=0.5
			self.loseDelta=5
		if episodeNumber==40000:
			self.winDelta=1
			self.loseDelta=10	
		return self.loseDelta, self.winDelta, self.alpha


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
		agent = WolfPHCAgent(learningRate = 0.2, discountFactor = 0.99, winDelta=0.001, loseDelta=0.01)
		agents.append(agent)

	numEpisodes = args.numEpisodes
	numTakenActions = 0
	goals=0
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
			if reward[0]==1:
				goals+=1
		print(episode,' : ',goals)
	print('Evala ',goals,' se 50000 epanalipseis epeidi eimai katsaplias')