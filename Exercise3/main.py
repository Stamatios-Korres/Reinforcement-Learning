#!/usr/bin/env python3
# encoding utf-8
from Environment import HFOEnv
import torch.multiprocessing as mp
from Networks import ValueNetwork

# Use this script to handle arguments and 
# initialize important components of your experiment.
# These might include important parameters for your experiment,
# your models, torch's multiprocessing methods, etc.

if __name__ == "__main__" :
	
	# Example on how to initialize global locks for processes
	# and counters.
	env = HFOEnv()
	env.connectToServer()
	env.startEnv()
	print("State returns:", env.act('MOVE'))
	counter = mp.Value('i', 0)
	lock = mp.Lock()
	processes = []
	for idx in range(0, args.num_processes):
		trainingArgs = (idx, args, value_network, target_value_network, optimizer, lock, counter)
		p = mp.Process(target=train, args=())
		p.start()
		processes.append(p)
	for p in processes:
		p.join()



