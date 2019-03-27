import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from Networks import ValueNetwork
from torch.autograd import Variable
from Environment import HFOEnv
import random
from goal import goal
import copy
import os
import numpy as np 

verbose=False

def change_lr(optimizer, counter):
    if counter > 4e6:
        lr = 1e-7
    elif counter > 3e6:
        lr = 1e-6
    elif counter > 1.5e6:
        lr = 5e-5
    else:
        lr = 1e-5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def hard_copy_a_to_b(valueNetwork, targetValueNetwork, counter, idx):
    # print('\n')
    # print('*'*100)
    # print('*'*40, 'UPDATING TARGET NETWORK', 'WRKER %d' %idx, counter, '*'*40)
    # print('*'*100)
    for target_param, param in zip(targetValueNetwork.parameters(), valueNetwork.parameters()):
                    target_param.data.copy_(param.data)
    print('UPDATE DONE')

def computeTargets(reward, nextState, discountFactor, done, targetNetwork):
    output_qs = targetNetwork(nextState)
    maxVal = torch.max(output_qs, dim=0)[0]
    target_val = reward + discountFactor * maxVal * (1 - done)
    return target_val

def computePrediction(state, action, value_network):
    output_qs = value_network(state)
    actValue = output_qs[action]# change dim when batch
    return actValue

def get_epsilon(episode, idx, counter, eval_mode):
    target_epsilon = np.arange(0.01,0.3,0.04)

    if eval_mode:
        epsilon = 0.0
    else:
        epsilon = 1.0 - ( (1 - target_epsilon[idx]) * min(counter/4000000, 1))
    return epsilon

def get_action(state, value_network, episode, idx, counter, eval_mode=False):
    output_qs = value_network(state)
    maxAction = torch.max(output_qs, dim=0)[1].item()
    if random.random() < get_epsilon(episode, idx, counter, eval_mode):      # e-greedy action
        maxAction = random.randint(0,3)
    return maxAction
    
# Function to save parameters of a neural network in pytorch.
def saveModelNetwork(model, strDirectory):
    torch.save(model.state_dict(), strDirectory)


def evaluate(value_network, hfoEnv, optimizer, max_episode_length, counter, idx, results_dir):
    f = open(os.path.join(results_dir, 'evaluation.out'), 'a')
    total_goals = 0.0
    total_steps = 0.0
    num_episodes= 100
    for episode in range(num_episodes):
        state = hfoEnv.reset()
        state_t = torch.Tensor(state)
        for step in range(max_episode_length):
            action_number = get_action(state_t, value_network, 0, 0, 0, eval_mode=True) # action from value networks
            action = hfoEnv.possibleActions[action_number]

            next_state, reward, done, status, info = hfoEnv.step(action, state)
            state_t = torch.Tensor(next_state)
            if done:
                break

        if status==1:
            total_steps += step
            total_goals += 1
        else:
            total_steps += max_episode_length
        state=next_state
    steps_per_goal = total_steps/num_episodes
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    epsilon = get_epsilon(None, idx, counter, eval_mode=False)
    f.write('Counter: %d\t Real goals: %d/%d\tSteps: %d\tSteps per episode: %.1f\tEpsilon: %.5f\tLR: %.2E\n' %(counter, total_goals, num_episodes, total_steps, steps_per_goal, epsilon, lr))
    f.flush()
    return steps_per_goal


def train(idx, args, value_network, target_value_network, optimizer, lock, counter, best_steps_per_goal, results_dir):

    port = 6000 + idx*100
    seed = idx*2
    if verbose:
        print("<><><><> " * 3, port)

    hfoEnv = HFOEnv(numTeammates=0, numOpponents=1, port=port, seed=seed)
    hfoEnv.connectToServer()

    f = open(os.path.join(results_dir, 'worker_%d.out'%idx), 'w')
    columns = '{0:<10} {1:<8} {2:<10} {3:<12} {4:<20} {5:<15} {6:<15}\n'
    f.write(columns.format('Episode','Status','Steps','Total steps','Avg steps to goal','Total Goals','Counter'))

    total_reward = 0
    total_goals = 0
    total_steps = 0
    local_step = 0
    for episode in range(args.num_episodes):
        # if verbose:
        #     print('\n','*'*100)
        #     print('*'*100)
        #     print('*'*40, 'EPISODE ', episode, '*'*40)
        #     print('*'*100)
        #     print('*'*100)

        done = False

        state = hfoEnv.reset()
        state_t = torch.Tensor(state)
        for step in range(args.max_episode_length):
            lock.acquire()
            counter.value += 1
            locked_counter = counter.value
            lock.release()
            local_step+=1

            # if verbose:
                # print('\n','*'*40, 'STEP ', step, '*'*40)

            action_number = get_action(state_t, value_network, episode, idx, counter.value, eval_mode=False) # action from value networks
            action = hfoEnv.possibleActions[action_number]

            next_state, reward, done, status, info = hfoEnv.step(action, state)
            next_state_t = torch.Tensor(next_state)
            total_reward += reward

            if verbose:
                print('STUFF' + 20*'*')
                print(action)
                print('Len(state): ', len(next_state),' Reward: ',reward,' Done: ', done, 'Status:', status, ' Info: ', info)
                print(next_state)


            # #### LEARN
            predicted_val = computePrediction(state_t, action_number, value_network)
            target_val = computeTargets(reward, next_state_t, args.gamma, done, target_value_network)

            # if verbose:
            #     print('predicted_val ', predicted_val)
            #     print('target_val', target_val)

            loss_function = nn.MSELoss()
            err = loss_function(predicted_val, target_val.detach())
            err.backward()


            # for param in value_network.parameters():
            #     param.grad.data.clamp_(-args.max_grads, args.max_grads)

            # # thelei with lock?
            if (local_step % args.value_update_steps == 0) or done:
                # print('Worker %d update grads at Step %d' % (idx, local_step))
                with lock:
                    clip_grad_norm_(value_network.parameters(), 0.5)
                    optimizer.step()
                    optimizer.zero_grad()

            grads_flag=False
            evaluate_flag=False
            if locked_counter % args.target_update_steps == 0:
                f.write('Update target network at counter %d\n' % locked_counter)
                hard_copy_a_to_b(value_network, target_value_network, locked_counter, idx)
            if (locked_counter % args.evaluate_freq_steps == 0):
                f.write('Evaluates value network at counter %d\n' % locked_counter)
                steps_per_goal = evaluate(copy.deepcopy(target_value_network), hfoEnv, optimizer, args.max_episode_length, locked_counter, idx, results_dir)
                if steps_per_goal <= best_steps_per_goal.value:
                    lock.acquire()
                    best_steps_per_goal.value = steps_per_goal
                    lock.release()
                    saveModelNetwork(value_network, os.path.join(results_dir, 'params_best'))
                    f2 = open(os.path.join(results_dir, 'evaluation.out'), 'a')
                    f2.write('Writing best params to file\n')
                    f2.close()
                # SAVE_EVERY=args.evaluate_freq_steps #
                change_lr(optimizer, counter)
                SAVE_EVERY=1000000
                if (locked_counter % SAVE_EVERY == 0):
                    saveModelNetwork(value_network, os.path.join(results_dir, 'params_%d' % int(locked_counter/SAVE_EVERY)))

            if done:
                break
            state = next_state
            state_t = next_state_t

        if status==1:
            total_steps += step
            total_goals += 1
        else:
            total_steps += 500

        f.write(columns.format(episode, status, step, total_steps, '%.1f'%(total_steps/(episode+1)), total_goals, locked_counter))
        f.flush()

    f.close()