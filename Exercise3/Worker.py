import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sys import maxsize
from Networks import ValueNetwork
from torch.autograd import Variable
from Environment import HFOEnv
import random
import numpy as np 

def init_environment(idx):
        hfoEnv = HFOEnv(numTeammates=0, numOpponents=1, port=6000+idx*4, seed=idx)
        hfoEnv.connectToServer()
        return hfoEnv

def set_epsilon(num_episode,idx):
	epsilons = [0.5,1,0.3,0.08,0.01,0.8,0,0.9]
	epsilon = epsilons[idx]
	epsilon = epsilon - num_episode/5000 
	if epsilon <0:
		epsilon = 0 
	return epsilon


def train(idx, val_network, target_value_network, optimizer, lock, counter,timesteps_per_process):
	
	# This runs a random agent

	episodeNumber = 0
	discountFactor = 0.99
	process_to_save_networks = 0
	goals = 0 
	locked_count = 0 
	
	
	asyc_update = 8
	copy_freq = 10000
	f = open('worker_%d.out'%idx, 'w')
	f2 = open('worker_saving%d.out'%idx, 'w')
	columns = '{0:<10} {1:<8} {2:<10} {3:<20} {4:<15} {5:<15}\n'
	f.write(columns.format('Episode','Status','Steps','Avg steps to goal','Total Goals','Counter'))

	hfoEnv = init_environment(idx)

	for local_timesteps in range(timesteps_per_process):
		done = False
		obs = hfoEnv.reset()
		timesteps = 0 
		while timesteps < 500:

			obs_tensor = torch.Tensor(obs).unsqueeze(0)
			_,act_number = compute_value_action(val_network, obs_tensor,idx,episodeNumber)
			act = hfoEnv.possibleActions[act_number]
			newObservation, reward, done, status, info = hfoEnv.step(act)

			Q_target = computeTargets(reward,torch.Tensor(newObservation).unsqueeze(0),discountFactor,done,target_value_network)
			Q = computePrediction(obs_tensor, act_number, val_network)
			loss_function = nn.MSELoss()
			loss = loss_function(Q_target,Q)
			loss.backward()
			with lock:
				counter.value +=1
				locked_count = counter.value
			if local_timesteps % asyc_update == 0:
				with lock:
					optimizer.step()
					optimizer.zero_grad()													
			timesteps+=1
			
			if done:
				break
		episodeNumber += 1
		if status ==1:
			goals+=1
		if(local_timesteps % copy_freq == 0):
			hard_copy(target_value_network, val_network)
		if locked_count % 100 == 0:
			f2.write('Process saving network')
			f2.flush()
				# saveModelNetwork(target_value_network,''+str(counter.value / 10**6)+"_model")
		f.write(columns.format(episodeNumber, status, local_timesteps, '%.1f'%(timesteps/(episodeNumber+1)), goals, counter.value))
		f.flush()
			

def compute_value_action(valueNetwork, obs,idx,episodeNumber):	
	
	output_qs = valueNetwork(obs)
	_,act =torch.max(output_qs[0],0)
	act = act.item()
	if random.random() < set_epsilon(episodeNumber,idx):
		act = random.randint(0,3)
	return output_qs[0][act],act

def computeTargets(reward, nextObservation, discountFactor, done, targetNetwork):
	
	if done:
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