#!/usr/bin/env python3
# encoding utf-8


from Environment import HFOEnv
import torch
import torch.multiprocessing as mp
from Networks import ValueNetwork
from Worker import train,hard_copy
import argparse
from SharedAdam import SharedAdam
import copy

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num_processes', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 2)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')

# Use this script to handle arguments and 
# initialize important components of your experiment.
# These might include important parameters for your experiment,
# your models, torch's multiprocessing methods, etc.

if __name__ == "__main__" :

	# Example on how to initialize global locks for processes
	# and counters.

	numOpponents = 1
	args = parser.parse_args()	

	# What is this instruction? Is it neccesary ? 
	mp.set_start_method('fork')

	counter = mp.Value('i', 0)
	lock = mp.Lock()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Hyperparameters - Features 
	features = 15
	actions = 4
	hidden_layers = [50,60,30]
	# Initialize the shared networks 
	
	val_network = ValueNetwork(features, hidden_layers, actions)
	val_network.share_memory()

	target_value_network = ValueNetwork(features, hidden_layers, actions)
	target_value_network.share_memory()


	hard_copy(val_network,target_value_network)
	val_network.train()
	


	# Shared optimizer ->  Share the gradients between the processes. Lazy allocation so gradients are not shared here
	optimizer = SharedAdam(params=val_network.parameters(),lr=1e-5)
	optimizer.share_memory()
	optimizer.zero_grad()
	timesteps_per_process = 32*(10**6) // args.num_processes*500

	processes = []
	for idx in range(0, args.num_processes):
			trainingArgs = (idx, val_network, target_value_network, optimizer, lock, counter,timesteps_per_process)
			p = mp.Process(target=train, args=trainingArgs)
			p.start()
			processes.append(p)
	for p in processes:
			p.join()

