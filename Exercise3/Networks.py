import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ValueNetwork(nn.Module):
    
  
    def __init__(self,inputDims = 15, layerDims = [50,40,30], outputDims = 4):

        super(ValueNetwork, self).__init__()

        self.processingLayers = []
        self.layerDims = layerDims.copy()
        self.layerDims.insert(0,inputDims)
        self.layerDims.append(outputDims)

        for idx in range(len(self.layerDims)-1):
            self.processingLayers.append(nn.Linear(self.layerDims[idx], self.layerDims[idx+1]))

        list_param = []
        for a in self.processingLayers:
            list_param.extend(list(a.parameters()))

        self.LayerParams = nn.ParameterList(list_param)

    def init_weights(self):
            torch.nn.init.xavier_uniform_(self.LayerParams)
                    
    def forward(self, inputs):
        out = inputs
        for layers in self.processingLayers[:-1]:
            out = layers(out)
            out = F.relu(out)

        out = self.processingLayers[-1](out)

        return out




