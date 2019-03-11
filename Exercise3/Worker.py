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
        hfoEnv = HFOEnv(numTeammates=0, numOpponents=1, port=6000+idx, seed=idx)
        hfoEnv.connectToServer()
        return hfoEnv

def train():
        pass

def computeTargets(reward, nextObservation, discountFactor, done, targetNetwork):
        pass

def computePrediction(state, action, valueNetwork):
        pass

# Function to save parameters of a neural network in pytorch.
def saveModelNetwork(model, strDirectory):
        torch.save(model.state_dict(), strDirectory)

# Set the target parameters to be exactly the same as the source
def hard_update(target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(param.data)

def implement_actor_critic(idx, target_value_network, optimizer, lock, counter):
        discount_factor = 0.9
        hfoEnv = init_environment(idx)
        episodeNumber = 0
        observation = hfoEnv.reset()
        while True:
                policy,stateValue = target_value_network(observation)
                act = hfoEnv.act(action)
                newObservation, reward, done, status, info = hfoEnv.step(act)
                print(newObservation, reward, done, status, info)

                if done:
                        episodeNumber += 1

