import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sys import maxsize
from Networks import ValueNetwork
from torch.autograd import Variable
from Environment import HFOEnv
import random

def init_environment(idx):
        hfoEnv = HFOEnv(numTeammates=0, numOpponents=1, port=6000+idx*5, seed=idx)
        hfoEnv.connectToServer()
        return hfoEnv

def train(idx, val_network, target_value_network, optimizer, lock, counter):
	
	# This runs a random agent

	episodeNumber = 0
	epsilon = 0.1
	hfoEnv = init_environment(idx)
	while True:
		done = False
		obs = hfoEnv.reset()
		while not done:
			obs_tensor = torch.Tensor(obs).unsqueeze(0)
			tens = compute_val(val_network, obs_tensor)
			
			act = torch.max(tens, dim=1)[1].item()
			if random.random() < epsilon:
				act = random.randint(0,3)
		
			act = hfoEnv.possibleActions[act]
			computePrediction(obs_tensor,act,val_network)
			newObservation, reward, done, status, info = hfoEnv.step(act)
			print(newObservation, reward, done, status, info)					
		if done:
			episodeNumber += 1

def computeTargets(reward, nextObservation, discountFactor, done, targetNetwork):
        pass

def computePrediction(state, action, valueNetwork):
        pass

def compute_val(value_network, obs):
	var_obs = Variable(obs)
	output_qs = value_network((var_obs))
	return output_qs 

# Function to save parameters of a neural network in pytorch.
def saveModelNetwork(model, strDirectory):
        torch.save(model.state_dict(), strDirectory)

# Set the target parameters to be exactly the same as the source
def hard_copy(target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(param.data)