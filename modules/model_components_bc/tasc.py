#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TaSc variants
"""

import torch
from torch import nn
import math 
import json
import numpy as np



with open('modules/config.txt', 'r') as f:
    args = json.load(f)
    
def operation(scaled_embeddings, approach, dim = -1):
    
    """
    defines the operation over the scaled embedding
    """
    
    assert approach in ["sum-over", "max-pool", "mean-pool"]
    
    if approach == "sum-over":
        
        return scaled_embeddings.sum(dim)
    
    elif approach == "max-pool":
        
        return scaled_embeddings.max(dim)[0]
    
    else:
        
        return scaled_embeddings.mean(dim)
        
        
class lin(nn.Module):
    
    def __init__(self, vocab_size, seed):
        
        super(lin, self).__init__()
        
        """
        Lin-TaSc where u is generated and multiplied with the embeddings
        to produce scaled non-contextualised embeddings
        
        operation over scaled embeddings to produce tasc scores s_i        
        """
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.vocab_size = vocab_size
        
        self.u = nn.Parameter(torch.randn(self.vocab_size))
        stdv = 1. / math.sqrt(self.u.size(0))
        self.u.data.uniform_(-stdv, stdv)
        
    def forward(self, sequence, masks, lengths, embeddings):
        
        tasc_weights = self.u[sequence.long()]
          
        tasc_weights.masked_fill_(masks.bool(), 0)
        
        tasc_weights = tasc_weights[:, :max(lengths)]
        
        scaled_embeddings = (tasc_weights.unsqueeze(-1) * embeddings[:,:max(lengths)])
        
        return operation(scaled_embeddings, args["operation"])
    
class conv(nn.Module):
    
    def __init__(self, vocab_size, seed, channel_dim = 15, kernel_dim = 1):
        
        super(conv, self).__init__()
        
        """
        Lin-TaSc where u is generated and multiplied with the embeddings
        to produce scaled non-contextualised embeddings
        
        convolution over scaled embeddings to produce tasc scores s_i        
        """
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.vocab_size = vocab_size
        self.channel_dim = 15
        self.kernel_dim = 1
        
        self.u = nn.Parameter(torch.randn(self.vocab_size))
        stdv = 1. / math.sqrt(self.u.size(0))
        self.u.data.uniform_(-stdv, stdv)
        
        self.conv_tasc = nn.Conv1d(args["embedding_dim"], channel_dim , kernel_dim)
        
    def forward(self, sequence, masks, lengths, embeddings):
        
        tasc_weights = self.u[sequence.long()]
          
        tasc_weights.masked_fill_(masks.bool(), 0)
        
        tasc_weights = tasc_weights[:, :max(lengths)]
        
        scaled_embeddings = (tasc_weights.unsqueeze(-1) * embeddings[:,:max(lengths)])
        
        filtered_e = self.conv_tasc(scaled_embeddings.transpose(1,2))

        return operation(filtered_e, args["operation"], dim=1)
    
    
    
class feat(nn.Module):
    
    def __init__(self, vocab_size, seed):
        
        super(feat, self).__init__()    
        
        """
        Lin-TaSc where U is generated and multiplied with the embeddings
        to produce scaled non-contextualised embeddings
        
        summation over embedding dimension to simulate dot-product to obtain s_i      
        """
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.vocab_size = vocab_size
    
        self.U = nn.Parameter(torch.randn([vocab_size, args["embedding_dim"]]))
        stdv = 1. / math.sqrt(self.U.size(0))
        self.U.data.uniform_(-stdv, stdv)
        
    def forward(self, sequence, masks, lengths, embeddings):   
        
        tasc_weights = self.U[sequence.long(), :]
                      
        tasc_weights.masked_fill_(masks.unsqueeze(-1).bool(), 0)
            
        tasc_weights = tasc_weights[:, :max(lengths)]
        
        scaled_embeddings = (embeddings[:,:max(lengths)] * tasc_weights)
        
        return operation(scaled_embeddings, args["operation"])