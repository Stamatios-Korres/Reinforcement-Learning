import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sys import maxsize
from Networks import ValueNetwork
from torch.autograd import Variable
from Environment import HFOEnv
import random
import copy
import numpy as np 

columns = '{0:<10} {1:<8} {2:<20} {3:<15} {4:<15}\n'

def init_environment(idx):
        hfoEnv = HFOEnv(numTeammates=0, numOpponents=1, port=6000+idx*10, seed=idx)
        hfoEnv.connectToServer()
        return hfoEnv

def set_epsilon(num_episode,idx,episode_per_process):
	init_epsilon = 1
	rate_of_decay = 10
	epsilon = init_epsilon - idx*(1/rate_of_decay)

	epsilon = epsilon - (epsilon)*num_episode/episode_per_process 
	if epsilon <0:
		epsilon = 0 
	return epsilon


def train(idx, val_network, target_value_network, optimizer, lock, counter,episode_per_process):
	
	# This runs a random agent

	
	discountFactor = 0.99
	goals = 0 
	total_steps = 0 
	asyc_update = 8
	copy_freq = 10000
	save_flag = False
	update_target = False
	evaluate_frequency = 300000
	f = open('worker_%d.out'%idx, 'w')
	

	evaluate_flag = False
	
	
	f.write(columns.format('Episode','Status','Avg steps to goal','Total Goals','Counter'))
	# f1.write(columns.format('Episode','Status','Avg steps to goal','Total Goals','Counter'))
	hfoEnv = init_environment(idx)

	for episode in range(episode_per_process):
		done = False
		state = hfoEnv.reset()
		timesteps = 0 
		for timesteps in range(500):

			obs_tensor = torch.Tensor(state).unsqueeze(0)
			Q,act_number = compute_value_action(val_network, obs_tensor,idx,episode,episode_per_process)
			act = hfoEnv.possibleActions[act_number]
			newObservation, reward, done, status, info = hfoEnv.step(act)

			Q_target = computeTargets(reward,torch.Tensor(newObservation).unsqueeze(0),discountFactor,done,target_value_network)
			# Q = computePrediction(obs_tensor, act_number, val_network)

			loss_function = nn.MSELoss()
			loss = loss_function(Q_target,Q)
			loss.backward()

			if timesteps % asyc_update == 0:
				with lock:
					optimizer.step()
					optimizer.zero_grad()	
			with lock:
				counter.value +=1	
				if counter.value % 1e6 == 0 :
					save_flag = True					
				if counter.value % copy_freq == 0:
					update_target = True	
				if counter.value % evaluate_frequency == 0:
					evaluate_flag = True
					
			if save_flag:
				saveModelNetwork(target_value_network,'params_'+str(int(counter.value/1e6)))
				save_flag = False
			if evaluate_flag:
				
				avg_goal_rate = evaluate(copy.deepcopy(target_value_network),hfoEnv,500,idx,episode_per_process,'worker_evaluate.out')
				evaluate_flag = False
				
			if update_target:
				hard_copy(target_value_network, val_network)
				update_target = False
			state = newObservation
			if done:
				break		
		if status ==1:
			total_steps+=timesteps
			goals+=1			
		else:
			total_steps+=500
		f.write(columns.format(episode, status, '%.1f'%(total_steps/(episode+1)), goals, counter.value))
		f.flush()
	
			

def compute_value_action(valueNetwork, obs,idx,episodeNumber,episode_per_process):	
	
	output_qs = valueNetwork(obs)
	_,act =torch.max(output_qs[0],0)
	act = act.item()
	if random.random() < set_epsilon(episodeNumber,idx,episode_per_process):
		act = random.randint(0,3)
	return output_qs[0][act],act

def computeTargets(reward, nextObservation, discountFactor, done, targetNetwork):
	
	if not done:
		output_qs = targetNetwork(nextObservation)
		target = reward + discountFactor*torch.max(output_qs[0],0)[0]
	else:
		target = torch.Tensor([reward])
	return target

def computePrediction(state, action, valueNetwork):
	output_qs = valueNetwork(state)
	return output_qs[0][action]

# Function to save parameters of a neural network in pytorch.
def saveModelNetwork(model, strDirectory):
        torch.save(model.state_dict(), strDirectory)

# Set the target parameters to be exactly the same as the source
def hard_copy(target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(param.data)

def evaluate(value_network, hfoEnv, max_episode_length, idx,episode_per_process,path):
	

	total_goals = 0.0
	total_steps = 0.0
	num_episodes= 25
	f1 = open(path, 'a')
	value_network.eval()
	for episode in range(num_episodes):
		state = hfoEnv.reset()
		
		for step in range(max_episode_length):
			obs_tensor = torch.Tensor(state).unsqueeze(0)
			_,act_number = compute_value_action(value_network, obs_tensor,idx,episode,3)
			act = hfoEnv.possibleActions[act_number]
			newObservation, reward, done, status, info = hfoEnv.step(act)
			state=newObservation
			if done:
				break

		if status==1:
			total_steps += step
			total_goals += 1
		else:
			total_steps += max_episode_length
	steps_per_goal = total_steps/num_episodes
	f1.write('Average Goals:'+str(steps_per_goal))
	return steps_per_goal