#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import itertools
import argparse
import numpy as np 
		
class JointQLearningAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, numTeammates, initVals=0.0):
		super(JointQLearningAgent, self).__init__()	
		
		self.discountFactor = discountFactor
		self.learningRate = learningRate
		self.actions = ['MOVE_UP', 'MOVE_DOWN', 'MOVE_LEFT', 'MOVE_RIGHT', 'KICK', 'NO_OP']
		self.numberOfActions = len(self.actions)
		self.numTeammates = numTeammates
		self.epsilon = epsilon
		
		# Initial empty lists 
		self.Q = {}
		self.C = {}
		self.N = {}	

		# Extra Information required for learning 

		self.S_t_1 = None
		self.S_t =   None
		self.a_t_1 = None
		self.opon_ac_t_1 = None
		self.r_t =   0


	
	def setExperience(self, state, action, oppoActions, reward, status, nextState):
	
		
		for Ai in self.actions:
			if state not in self.C.keys():
				self.C[(state,Ai)]=0
			if nextState != (-1,-1):
				self.N[nextState]=0.0
				if nextState not in self.C.keys():
					if nextState not in self.C.keys():
						for Aj in self.actions:
							self.Q[(nextState,(Ai,Aj))]=0.0
					
		self.S_t_1 = state
		# Hard coded the number of oponnets to one 
		self.opon_ac_t_1 = oppoActions[0]
		self.a_t_1 = action
		self.S_t = nextState
		self.r_t = reward

	def learn(self):
		prior = self.Q[(self.S_t_1,(self.a_t_1,self.opon_ac_t_1))]
		if self.S_t != (-1, -1):
			self.Q[(self.S_t_1,(self.a_t_1,self.opon_ac_t_1))] = (1 - self.learningRate) * self.Q[(self.S_t_1,(self.a_t_1,self.opon_ac_t_1))] \
				+ self.learningRate*(self.r_t +  self.discountFactor*(self.get_max_action_value_new_state()) )
			self.C[(self.S_t_1,self.opon_ac_t_1)] +=1
			self.N[self.S_t_1]+=1
		else:
			self.Q[(self.S_t_1,(self.a_t_1,self.opon_ac_t_1))] = (1 - self.learningRate) * self.Q[(self.S_t_1,(self.a_t_1,self.opon_ac_t_1))] \
				+ self.learningRate*self.r_t
			self.C[(self.S_t_1,self.opon_ac_t_1)] +=1
			self.N[self.S_t_1]+=1
		return self.Q[(self.S_t_1,(self.a_t_1,self.opon_ac_t_1))] - prior
	
	
	def get_max_action_value(self):
		joint_max_sum = []
		for Ai in self.actions:
			sumAi = 0 
			if self.N[self.S_t] != 0:
				for Aj in self.actions:
					sumAi += self.C[(self.S_t,Aj)] * self.Q[(self.S_t,(Ai,Aj))] / self.N[self.S_t]
			joint_max_sum.append(sumAi)
		index = joint_max_sum.index(max(joint_max_sum))
		if joint_max_sum[1:] == joint_max_sum[:-1]:
			index = np.random.randint(0,self.numberOfActions)
		return self.actions[index]

	def get_max_action_value_new_state(self):
		joint_max_sum = []
		for Ai in self.actions:
			sumAi = 0 
			if self.N[self.S_t] != 0:
				for Aj in self.actions:
					sumAi += self.C[(self.S_t,Aj)] * self.Q[(self.S_t,(Ai,Aj))] / self.N[self.S_t]
			joint_max_sum.append(sumAi)		
		
		return max(joint_max_sum)

	


	def setState(self, state):
		if state not in self.N.keys():
			self.N[state]=0
			for Ai in self.actions:
				print((state,Ai))
				self.C[(state,Ai)]=0
				for Aj in self.actions:
					self.Q[(state,(Ai,Aj))]=0.0
		self.S_t = state 



	def act(self):		
		return self.get_max_action_value()

		

	def setEpsilon(self, epsilon) :
		self.epsilon = epsilon
		
	def setLearningRate(self, learningRate) :
		self.learningRate = learningRate

	
	def toStateRepresentation(self, state):
			if type(state) == str:
				return -1, -1
			else:
				attacker1 = tuple(state[0][0])
				attacker2 = tuple(state[0][1])
				ball = tuple(state[1][0])
				defender = tuple(state[2][0])
			return (attacker1,attacker2,ball,defender)

	def computeHyperparameters(self, numTakenActions, episodeNumber):
		learning = max(1-episodeNumber/45500,1e-5)
		e = 0.1
		if episodeNumber > 30000:
			e = 0.005
		elif episodeNumber > 45000:
			e = 1e-5
		return learning,e
		
if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--numOpponents', type=int, default=1)
	parser.add_argument('--numAgents', type=int, default=2)
	parser.add_argument('--numEpisodes', type=int, default=50000)

	args=parser.parse_args()

	MARLEnv = DiscreteMARLEnvironment(numOpponents = args.numOpponents, numAgents = args.numAgents)
	agents = []
	numAgents = args.numAgents
	numEpisodes = args.numEpisodes
	for i in range(numAgents):
		agent = JointQLearningAgent(learningRate = 0.1, discountFactor = 0.9, epsilon = 1.0, numTeammates=args.numAgents-1)
		agents.append(agent)

	numEpisodes = numEpisodes
	numTakenActions = 0
	final_goals = 0
	total_goals = 0  
	for episode in range(numEpisodes):	
		status = ["IN_GAME","IN_GAME","IN_GAME"]
		observation = MARLEnv.reset()
			
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
				agents[agentIdx].setState(agents[agentIdx].toStateRepresentation(obsCopy))
				actions.append(agents[agentIdx].act())

			nextObservation, reward, done, status = MARLEnv.step(actions)
			numTakenActions += 1

			for agentIdx in range(args.numAgents):
				oppoActions = actions.copy()
				del oppoActions[agentIdx]
				agents[agentIdx].setExperience(agents[agentIdx].toStateRepresentation(stateCopies[agentIdx]), actions[agentIdx], oppoActions, 
					reward[agentIdx], status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
				agents[agentIdx].learn()

			if reward[0] ==1:
				total_goals +=1
			print(episode,total_goals)
			if episode > 44999:
				if reward[0] == 1:
					final_goals +=1
				
			observation = nextObservation
	print(episode,total_goals)
	print(episode,final_goals)
				
