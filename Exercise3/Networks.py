import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gym

#  self.number_of_actions = 2
#         self.gamma = 0.99
#         self.final_epsilon = 0.0001
#         self.initial_epsilon = 0.1
#         self.number_of_iterations = 2000000
#         self.replay_memory_size = 10000
#         self.minibatch_size = 32

# Define your neural networks in this class. 
# Use the __init__ method to define the architecture of the network
# and define the computations for the forward pass in the forward method.

# class ValueNetwork(nn.Module):

#         def init_weights(self,m):
#                 if type(m) == nn.Linear:
#                         torch.nn.init.xavier_uniform_(m.weight)
#                         m.bias.data.fill_(0.01)

#         # I implemented only an easy linear baseline network
#         def __init__(self,state_size, action_size, hidden_dimension):
#                 super(ValueNetwork, self).__init__()
#                 self.fc1 = nn.Linear(state_size+1, hidden_dimension)
#                 self.fc2 = nn.Linear(hidden_dimension, action_size)
#                 self.Smax = nn.Softmax()
#                 self.fc3 = nn.Linear(25, action_size)


#         def forward(self, x):
#                 x = torch.Tensor(x)
#                 out = F.relu(self.fc1(x))
#                 out = F.relu(self.fc2(out))
#                 policy = self.Smax(out)
#                 out = self.fc3(policy)
#                 return policy, out


# 				import torch.nn as nn


class ValueNetwork(nn.Module):
    def __init__(self,inputDims, layerDims, outputDims):

        super(ValueNetwork, self).__init__()

        self.processingLayers = []
        self.layerDims = layerDims
        self.layerDims.insert(0,inputDims)
        self.layerDims.append(outputDims)

        for idx in range(len(self.layerDims)-1):
            self.processingLayers.append(nn.Linear(self.layerDims[idx], self.layerDims[idx+1]))

        list_param = []
        for a in self.processingLayers:
            list_param.extend(list(a.parameters()))

        self.LayerParams = nn.ParameterList(list_param)

    def forward(self, inputs):

        out = inputs
        for layers in self.processingLayers[:-1]:
            out = layers(out)
            out = F.relu(out)

        out = self.processingLayers[-1](out)

        return out




