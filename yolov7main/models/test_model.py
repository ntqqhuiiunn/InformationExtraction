import numpy as np
import random
import torch
import torch.nn as nn

class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output

def main(weights):
    model = Ensemble()
    cpkt = torch.load()
if __name__ == '__main__':
    weights = "./weights/best.pt"
    main(weights)