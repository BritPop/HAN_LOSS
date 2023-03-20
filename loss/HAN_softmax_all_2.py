#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Adapted from https://github.com/CoinCheung/pytorch-loss (MIT License)

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from utils import accuracy

class LossFunction(nn.Module):
    def __init__(self, num_out, num_class, scale=15, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        
        self.s = scale
        self.in_feats = num_out
        self.W = torch.nn.Parameter(torch.randn(num_out, num_class), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)
        self.nClasses = num_class
        print('Initialised HAN Softmax ALL')

    def forward(self, x, label=None):

        if x.size()[1] == self.in_feats:

            x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            x_norm = torch.div(x, x_norm)
            w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
            w_norm = torch.div(self.W, w_norm)

            costh = torch.mm(x_norm, w_norm) # (input, n_class)
            costh_w = torch.mm(w_norm.transpose(0,1), w_norm) # (n_class, n_class)
            #costh_w = F.cosine_similarity(w_norm.transpose(0,1).unsqueeze(-1),w_norm.transpose(0,1).unsqueeze(-1).transpose(0,2))
            costh_x = torch.mm(x_norm, x_norm.transpose(0,1)) # (input, input)
            #costh_x = F.cosine_similarity(x.unsqueeze(-1),x.unsqueeze(-1).transpose(0,2))

            stepsize = costh_w.size()[0]
            s_w_n = costh_w.flatten()[1:].view(stepsize-1, stepsize+1)[:,:-1].transpose(0,1).flatten()[0:stepsize*(stepsize-1)//2]

            stepsize = costh_x.size()[0]
            s_x_n = costh_x.flatten()[1:].view(stepsize-1, stepsize+1)[:,:-1].transpose(0,1).flatten()[0:stepsize*(stepsize-1)//2]

            n_rows, n_cols = costh.shape
            shifts = -1 * label.unsqueeze(-1)
            arange1 = torch.arange(n_cols).view(( 1,n_cols)).repeat((n_rows,1)).cuda()
            arange2 = (arange1 - shifts) % n_cols
            costh2 = torch.gather(costh, 1, arange2)

            s_p = costh2[:,0]
            s_n = costh2[:,1:].flatten()

            s_n_a = torch.cat([s_n, s_w_n, s_x_n])
            s_n_a, _ = s_n_a.topk(10000)

            s_p = s_p.unsqueeze(-1) 
            min_s_p = torch.min(s_p)
            check = torch.where(s_n_a>min_s_p, True, False)
            s_n_a = torch.masked_select(s_n_a, check)

            s_n_a = s_n_a.unsqueeze(0).repeat(n_rows, 1) #s_p, s_n

            s = torch.cat([s_p, s_n_a], dim=-1)
            cos_sim_matrix2 = s*self.s

            label2 = torch.from_numpy(numpy.zeros(n_rows).astype(int)).cuda()

            costh_s = self.s * costh

            loss    = self.ce(cos_sim_matrix2, label2)

            prec1   = accuracy(costh_s.detach(), label.detach(), topk=(1,))[0]
            prec2   = accuracy(cos_sim_matrix2.detach(), label2.detach(), topk=(1,))[0]

        else:
            
            x_q_norm = torch.norm(x[:,0,:], p=2, dim=1, keepdim=True).clamp(min=1e-12)
            x_q_norm = torch.div(x[:,0,:], x_q_norm)
            
            x_s_norm = torch.norm(torch.mean(x[:,1:,:],1), p=2, dim=1, keepdim=True).clamp(min=1e-12)
            x_s_norm = torch.div(torch.mean(x[:,1:,:],1), x_s_norm)
            
            w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
            w_norm = torch.div(self.W, w_norm)

            costh_q_w = torch.mm(x_q_norm, w_norm) # (input, n_class)
            costh_s_w = torch.mm(x_s_norm, w_norm) # (input, n_class)
            costh_q_s = torch.mm(x_q_norm, x_s_norm.transpose(0,1)) 
            
            costh_w = torch.mm(w_norm.transpose(0,1), w_norm) # (n_class, n_class)
            #costh_w = F.cosine_similarity(w_norm.transpose(0,1).unsqueeze(-1),w_norm.transpose(0,1).unsqueeze(-1).transpose(0,2))
            costh_x_q = torch.mm(x_q_norm, x_q_norm.transpose(0,1)) # (input, input)
            #costh_x = F.cosine_similarity(x.unsqueeze(-1),x.unsqueeze(-1).transpose(0,2))
            costh_x_s = torch.mm(x_s_norm, x_s_norm.transpose(0,1)) # (input, input)

            n_rows, n_cols = costh_q_w.shape
            shifts = -1 * label.unsqueeze(-1)
            arange1 = torch.arange(n_cols).view(( 1,n_cols)).repeat((n_rows,1)).cuda()
            arange2 = (arange1 - shifts) % n_cols
            
            costh2_q_w = torch.gather(costh_q_w, 1, arange2)
            costh2_s_w = torch.gather(costh_s_w, 1, arange2)
            
            s_q_w_p = costh2_q_w[:,0]
            s_s_w_p = costh2_s_w[:,0]
            s_q_w_n = costh2_q_w[:,1:].flatten()
            s_s_w_n = costh2_s_w[:,1:].flatten()
            
            stepsize = n_rows
            s_q_s_p = costh_q_s.diagonal()  
            s_q_s_n = costh_q_s.flatten()[1:].view(stepsize-1, stepsize+1)[:,:-1].flatten() #stepsize*stepsize-1
            
            stepsize = n_cols
            s_w_n = costh_w.flatten()[1:].view(stepsize-1, stepsize+1)[:,:-1].transpose(0,1).flatten()[0:stepsize*(stepsize-1)//2]

            stepsize = n_rows
            s_x_q_n = costh_x_q.flatten()[1:].view(stepsize-1, stepsize+1)[:,:-1].transpose(0,1).flatten()[0:stepsize*(stepsize-1)//2]
            s_x_s_n = costh_x_s.flatten()[1:].view(stepsize-1, stepsize+1)[:,:-1].transpose(0,1).flatten()[0:stepsize*(stepsize-1)//2]

            
            s_n = torch.cat([s_q_s_n, s_q_w_n, s_s_w_n, s_x_q_n, s_x_s_n, s_w_n])
            s_n, _ = s_n.topk(10000)
            
            s_p = torch.cat([s_q_s_p, s_q_w_p, s_s_w_p])
            s_p = s_p.unsqueeze(-1)
            min_s_p = torch.min(s_p)
            check = torch.where(s_n>min_s_p, True, False)
            s_n = torch.masked_select(s_n, check)

            bs = s_p.shape[0]
            s_n = s_n.unsqueeze(0).repeat(bs, 1) 

            s = torch.cat([s_p, s_n], dim=-1)            
            
            cos_sim_matrix2 = s*self.s
            label2 = torch.from_numpy(numpy.zeros(bs).astype(int)).cuda()

            costh_q_w = self.s * costh_q_w

            loss    = self.ce(cos_sim_matrix2, label2)

            prec1   = accuracy(costh_q_w.detach(), label.detach(), topk=(1,))[0]
            prec2   = accuracy(cos_sim_matrix2.detach(), label2.detach(), topk=(1,))[0]

        return loss, prec1, prec2
