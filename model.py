"""
Created on Thu Feb 27 21:13:07 2020

@author: rong
"""

import torch
import torch.nn as nn
import numpy as np
import quadprog

from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d, MetaLinear)
from torchmeta.modules.utils import get_subdict

def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers => 各個layer的數量的list
        tid: task id
    """
    grads[:, tid].fill_(0.0)
    cnt = 0
    
    # grad 是parameter的attr
    for param in pp():
        if param.grad is not None:
            # beg = begginning
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[tid, beg: en].copy_(param.grad.data.view(-1))
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.
        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))

def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1
        
        
def conv3x3(in_channels, out_channels, **kwargs):
    return MetaSequential(
        MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        MetaBatchNorm2d(out_channels, momentum=1., track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class ConvolutionalNeuralNetwork(MetaModule):
    def __init__(self, in_channels, out_features, args, hidden_size=64):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.overwrite_num = 0
        self.margin = 0  
        self.old_task =-1
        self.observed_tasks = []
        self.memory_data_spt = torch.FloatTensor(2, args.batch_size, args.num_ways * args.num_shots, in_channels, 84, 84).cuda()
        self.memory_labs_spt = torch.LongTensor(2, args.batch_size, args.num_ways * args.num_shots).cuda()
        self.memory_data_qry = torch.FloatTensor(2, args.batch_size, 75, in_channels, 84, 84).cuda()
        self.memory_labs_qry = torch.LongTensor(2, args.batch_size, 75).cuda()
        self.grad_dims = [864, 32, 32, 32, 9216, 32, 32, 32, 9216, 32, 32, 32, 9216, 32, 32, 32, 4000, 5] 

        self.grads = torch.Tensor(2, sum(self.grad_dims))
        self.ce = nn.CrossEntropyLoss()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size

        self.features = MetaSequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size)
        )
        self.classifier = MetaLinear(hidden_size * 5 * 5, out_features)

    def forward(self, inputs, params=None):
                
        features = self.features(inputs, params=get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        logits = self.classifier(features, params=get_subdict(params, 'classifier'))
        return logits