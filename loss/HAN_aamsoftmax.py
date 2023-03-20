#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Adapted from https://github.com/wujiyang/Face_Pytorch (Apache License)

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy, math
from utils import accuracy

class LossFunction(nn.Module):
    def __init__(self, num_out, num_class, margin=0.3, scale=15, easy_margin=False, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        
        self.m = margin
        self.s = scale
        self.in_feats = num_out
        self.weight = torch.nn.Parameter(torch.FloatTensor(num_class, num_out), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

        print('Initialised AAMSoftmax margin %.3f scale %.3f'%(self.m,self.s))

    def forward(self, x, label=None):

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats
        
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
    
        n_rows, n_cols = output.shape
        shifts = -1 * label.unsqueeze(-1)
        arange1 = torch.arange(n_cols).view(( 1,n_cols)).repeat((n_rows,1)).cuda()
        arange2 = (arange1 - shifts) % n_cols
        output2 = torch.gather(output, 1, arange2)
        
        s_p = output2[:,0]
        s_n = output2[:,1:].flatten()
        s_p = s_p.unsqueeze(-1) 
        s_n = s_n.unsqueeze(0).repeat(n_rows, 1) #s_p, s_n
        
        s = torch.cat([s_p, s_n], dim=-1)
        label2 = torch.from_numpy(numpy.zeros(n_rows).astype(int)).cuda()
        cos_sim_matrix2 = s*self.s
        output = output * self.s
        loss    = self.ce(cos_sim_matrix2, label2)
        prec1   = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        prec2   = accuracy(cos_sim_matrix2.detach(), label2.detach(), topk=(1,))[0]
        
        return loss, prec1, prec2
