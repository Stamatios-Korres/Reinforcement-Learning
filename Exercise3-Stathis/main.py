# Try value update steps 8
# Try target network update steps 5000
# Try clip_grad_norm_ https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/language_model/main.py
# Try changing learning rate in SharedAdam
# Try other stuff in rewards

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
import os

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

parser.add_argument('--value-update-steps', type=int, default=8, help='number of forward steps in DQN before the value networks update (default: 5)') #8
parser.add_argument('--target-update-steps', type=int, default=5000, help='number steps before the target network update (default: 20000)') # try 5000
parser.add_argument('--evaluate-freq-steps', type=int, default=200000, help='number of steps for evaluation 100.000')

parser.add_argument('--max-episode-length', type=int, default=500, help='maximum length of an episode (default: 500)')
parser.add_argument('--num-episodes', type=int, default=50000, help='number of episodes (default: 20000)')

if __name__ == "__main__" :
    args = parser.parse_args()
    
    for i in range(1000):
        results_dir = './results_%04d' %i
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            break


    f = open(os.path.join(results_dir, 'evaluation.out'), 'w')
    f.close()
    # Example on how to initialize global locks for processes
    mp.set_start_method('spawn')
    # and counmp.Processters.
    counter = mp.Value('i', 0)
    best_steps_per_goal = mp.Value('f', 500.0)
    lock = mp.Lock()

    # Features 
    N_LOW_LEVEL_FEATURES = 68
    N_HIGH_LEVEL_FEATURES = 15
    N_ACTIONS = 4

    #### HYPER PARAMETERS
    N_FEATURES= N_HIGH_LEVEL_FEATURES

    # LAYERS = [90,90,90]
    value_network = ValueNetwork()
    target_value_network = ValueNetwork()
    hard_copy_a_to_b(value_network, target_value_network, -1, -1)
    target_value_network.share_memory()
    value_network.share_memory()
    value_network.train()

    optimizer = SharedAdam(value_network.parameters(), lr=args.lr)
    optimizer.share_memory()
    optimizer.zero_grad()

    processes=[]
    for idx in range(0, args.num_processes):
        trainingArgs = (idx, args, value_network, target_value_network, optimizer, lock, counter, best_steps_per_goal, results_dir)
        p = mp.Process(target=train, args=trainingArgs)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print(counter.value)