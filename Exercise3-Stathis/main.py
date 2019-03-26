#!/usr/bin/env python3
# encoding utf-8
from Environment import HFOEnv
import random
import argparse
import torch.multiprocessing as mp
import torch
from torch import Tensor
from SharedAdam import SharedAdam
from Worker import train, hard_copy_a_to_b 
import argparse
from Networks import ValueNetwork

# Use this script to handle arguments and 
# initialize important components of your experiment.
# These might include important parameters for your experiment, and initialization of
# your models, torch's multiprocessing methods, etc.

parser = argparse.ArgumentParser(description='ADQN')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
parser.add_argument('--max-grads', type=float, default=1, help='max gradient (default: 1)')
# parser.add_argument('--tau', type=float, default=1.00, help='parameter for GAE (default: 1.00)')
# parser.add_argument('--entropy-coef', type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
# parser.add_argument('--value-loss-coef', type=float, default=0.5, help='value loss coefficient (default: 0.5)')
parser.add_argument('--num-processes', type=int, default=5, help='how many training processes to use (default: 5)')

parser.add_argument('--value-update-steps', type=int, default=5, help='number of forward steps in DQN before the value networks update (default: 5)')
parser.add_argument('--target-update-steps', type=int, default=20000, help='number steps before the target network update (default: 20000)')
parser.add_argument('--evaluate-freq-steps', type=int, default=200000, help='number of steps for evaluation 100.000')

parser.add_argument('--max-episode-length', type=int, default=500, help='maximum length of an episode (default: 500)')
parser.add_argument('--num-episodes', type=int, default=50000, help='number of episodes (default: 20000)')

if __name__ == "__main__" :
    args = parser.parse_args()
    
    f = open('evaluation.out', 'w')
    f.close()
    # Example on how to initialize global locks for processes
    mp.set_start_method('fork')
    # and counmp.Processters.
    counter = mp.Value('i', 0)
    lock = mp.Lock()

    # Features 
    N_LOW_LEVEL_FEATURES = 68
    N_HIGH_LEVEL_FEATURES = 15
    N_ACTIONS = 4

    #### HYPER PARAMETERS
    N_FEATURES= N_HIGH_LEVEL_FEATURES
    HIDDEN_LAYERS = [30,30,30]

    value_network = ValueNetwork(N_FEATURES, HIDDEN_LAYERS, N_ACTIONS)
    target_value_network = ValueNetwork(N_FEATURES, HIDDEN_LAYERS, N_ACTIONS)
    hard_copy_a_to_b(value_network, target_value_network, -1, -1)
    target_value_network.share_memory()
    value_network.share_memory()
    value_network.train()

    optimizer = SharedAdam(value_network.parameters(), lr=args.lr)
    optimizer.share_memory()
    optimizer.zero_grad()

    processes=[]
    for idx in range(0, args.num_processes):
        trainingArgs = (idx, args, value_network, target_value_network, optimizer, lock, counter)
        p = mp.Process(target=train, args=trainingArgs)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


    print(counter.value)
