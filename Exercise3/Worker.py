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

def train(idx, val_network, target_value_network, optimizer, lock, counter,timesteps_per_process):
	
	# This runs a random agent

	episodeNumber = 0
	discountFactor = 0.99
	process_to_save_networks = 0
	
	
	hfoEnv = init_environment(idx)
	local_timesteps = 0
	asyc_update = 10
	copy_freq = 1000
	epislon = [0.1,0.01,0.5]
	while local_timesteps < timesteps_per_process:
		done = False
		obs = hfoEnv.reset()
		timesteps = 0 
		while timesteps<500:

			obs_tensor = torch.Tensor(obs).unsqueeze(0)
			Q,act = compute_val(val_network, obs_tensor,idx,epislon)
			act = hfoEnv.possibleActions[act]
			newObservation, reward, done, status, info = hfoEnv.step(act)

			Q_target = computeTargets(reward,torch.Tensor(newObservation).unsqueeze(0),discountFactor,done,target_value_network)

			loss_function = nn.MSELoss()
			loss = loss_function(Q_target,Q)
			loss.backward()

			if local_timesteps % asyc_update == 0:
				with lock:
					optimizer.step()
					optimizer.zero_grad()			
			
			# print(newObservation, reward, done, status, info)	
			with lock:
				counter.value +=1
			timesteps+=1
			local_timesteps+=1				
			
		if done:
			episodeNumber += 1
		if(local_timesteps % copy_freq == 0):
			hard_copy(target_value_network, val_network)
		if (counter.value % 10**6) == 0:
			if idx == process_to_save_networks:
				saveModelNetwork(target_value_network,'/afs/inf.ed.ac.uk/user/s18/s1877727/HFO/example/RL2019-BaseCodes/Exercise3'+str(counter.value / 10**6)+"_model")
			# print(" I have to update the target network")


		
		

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

	

def compute_val(valueNetwork, obs,idx,epsilon):
	
	probabilities = [0.4,0.3,0.3]
	e = np.random.choice(epsilon,size=1,p=probabilities)
	
	# epsilon = [0.5,0.1,0.2,0.05,0.001,1,1e-8,0.25]
	output_qs = valueNetwork(obs)
	
	_,act =torch.max(output_qs[0],0)
	act = act.item()
	
	if random.random() < e:
		act = random.randint(0,3)
	return output_qs[0][act],act

# Function to save parameters of a neural network in pytorch.
def saveModelNetwork(model, strDirectory):
        torch.save(model.state_dict(), strDirectory)

# Set the target parameters to be exactly the same as the source
def hard_copy(target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(param.data)