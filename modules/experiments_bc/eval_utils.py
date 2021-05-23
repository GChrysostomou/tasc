import torch
import torch.nn as nn
import numpy as np
import scipy.stats as scistats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statistics
from torch.autograd import Variable
import pickle
import pandas as pd
import random
from sklearn.metrics import *
from sklearn.metrics import precision_recall_fscore_support as prfs

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def stat(results, name , sig_thresh = 0.33):
    
    """
    input : results as tuple (cors, sig_fracs)
    """

    taus = [float(x) for x in results[0]]
    sig_frac = results[1]

    stats = {}

    stats["mean"] = sum(taus)/len(taus)

    stats["stdev"] = statistics.stdev(taus)

    stats["sig_frac"] = len(sig_frac) / len(taus)
    
    stats["name"] = name

    return stats


def plotting(results_neg, results_pos, save_name, titles, xlab, xlim = False):
    
    plt.figure(figsize=(12,8))

    taus_neg = results_neg[0]
    taus_pos = results_pos[0]

    plt.hist(taus_neg, bins = 30, alpha = 0.7, lw = 1, color= 'b', edgecolor = "black", label = "Negative")
    plt.hist(taus_pos, bins = 30, alpha = 0.7, lw = 1, color = "g", edgecolor = "black", label = "Positive")
    plt.title(titles)
    
    plt.xlabel(xlab) 
    plt.ylabel("Counts")
    plt.legend(loc = "upper left")

    if xlim:

        plt.xlim(left = -1, right = 1)

    plt.savefig(save_name + ".png", bbox_inches='tight')
        
    plt.clf()
    plt.close()
    
    if len(results_neg[0]) == 0:

        results_neg = [[0,0],[0,0]]

    if len(results_pos[0]) == 0:

        results_pos = [[0,0],[0,0]]

    stats_taus = [stat(results_neg, name = "neg"), stat(results_pos, name = "pos")]

    csv_columns = stats_taus[0].keys()

    import csv
    with open(save_name + ".csv", 'w') as csvfile:

        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)

        writer.writeheader()

        for data in stats_taus:

            writer.writerow(data)



def grad_checkers(model, switch = "off"):
    
    if switch == "off":
        for n,p in model.named_parameters():

            a = n.split(".")

            if a[0] == "attention":

                p.requires_grad = False

            if a[0] == "embedding":

                p.requires_grad = True
                
                
    elif switch == "all":
        
        for n,p in model.named_parameters():
            
            p.requires_grad = True

    else:
        
        for n,p in model.named_parameters():

            a = n.split(".")

            if a[0] == "attention":

                p.requires_grad = True

            if a[0] == "embedding":

                p.requires_grad = True




def tvd(a, b):

    return (torch.abs(a.float()-b.float())).sum(-1)/2


def maxim(tensor):

    _, idx = torch.max(tensor, dim=-1)

    return idx

def kld(a1, a2):

    a1 = torch.clamp(a1, 0, 1)
    a2 = torch.clamp(a2, 0, 1)
    log_a1 = torch.log2(a1 + 1e-10)
    log_a2 = torch.log2(a2 + 1e-10)
    kld = a1 * (log_a1 - log_a2)
    kld = kld.sum(-1)

    return kld


def jsd(p,q):

    m = (p + q) * 0.5

    return 0.5*(kld(p,m) + kld(q,m))

def generate_uniform_attn(sentence, lengths) :
        attn = np.zeros((sentence.shape[0], sentence.shape[1]))
        inv_l = 1. / lengths.cpu().data.numpy()
        attn += inv_l[:, None]
        return torch.Tensor(attn).to(device)

