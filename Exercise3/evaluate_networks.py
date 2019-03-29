from sys import path
from os.path import dirname as dir
path.append(dir(path[0]))

from Environment import HFOEnv
import random
import torch
from torch import Tensor
from Networks import ValueNetwork
import os
import json
from Worker import compute_value_action

def evaluate(value_network, hfoEnv, max_episode_length, results_dir):
    print('results dir', results_dir)
    f = open(results_dir, 'w')
    total_goals = 0.0
    total_steps = 0.0
    num_episodes= 500
    for episode in range(num_episodes):
        state = hfoEnv.reset()
        
        for step in range(max_episode_length):

            obs_tensor = torch.Tensor(state).unsqueeze(0)
            Q, act = compute_value_action(value_network, obs_tensor,0,0,max_episode_length)
            act = hfoEnv.possibleActions[act]
            next_state, reward, done, status, info = hfoEnv.step(act)
            #_, action_number = compute_val(value_network, state_t, 0, 0) # action from value networks
            #action = hfoEnv.possibleActions[action_number]

            #next_state, reward, done, status, info = hfoEnv.step(action)
            
            if done:
                break
            state=next_state
        if status==1:
            total_steps += step
            total_goals += 1
        else:
            total_steps += max_episode_length
        
    steps_per_goal = total_steps/num_episodes
    if (episode % 10) == 0:
        print('Episode %d\tReal goals: %d/%d\tSteps: %d\tSteps per episode: %.1f' %(episode, total_goals, num_episodes, total_steps, steps_per_goal))

    f.write('Real goals: %d/%d\tSteps: %d\tSteps per episode: %.1f\n' %(total_goals, num_episodes, total_steps, steps_per_goal))
    f.flush()
    f.close()


value_network = ValueNetwork(15,[50,40,30],4)

idx = 12
port = 8000 + idx*2
seed = idx*2



print('starting HFO')
hfoEnv = HFOEnv(numTeammates=0, numOpponents=1, port=port, seed=seed)
hfoEnv.connectToServer()

for d in range(1,7):
    fn = 'params_'+str(d+1)
    print('Loading from ', fn)
    value_network.load_state_dict(torch.load(fn))
    evaluate(value_network, hfoEnv, 500, 'results_'+str(d+1))

#import os
#directories = [x[0] for x in os.walk('.') if 'results' in x[0]]

#for d in directories:
#    fn = os.path.join(d, 'evaluation_best.out')
#    f = open(fn,'r')
#    print(d)
#    for ln in f:
#        print(ln)
