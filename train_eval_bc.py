#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import os, sys
import numpy as np
import pandas as pd
import argparse
from modules.utils import dataholder, dataholder_bert
import json

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



parser = argparse.ArgumentParser()

parser.add_argument("-dataset", type = str, help = "select dataset / task", default = "sst")
parser.add_argument("-encoder", type = str, help = "select encoder", default = "lstm", choices = ["lstm", "gru", "mlp", "cnn", "bert"])
parser.add_argument("-data_dir", type = str, help = "directory of saved processed data", default = "data/")
parser.add_argument("-model_dir", type = str, help = "directory to save models", default = "test_models/")
parser.add_argument("-experiments_dir", type = str, help = "directory to save models", default = "test_experiment_results/")
parser.add_argument("-mechanism", type = str, help = "choose mechanism", default = "tanh", choices = ["tanh", "dot"] )
parser.add_argument('-operation',  type = str, help='operation over scaled embedding', 
                    default='sum-over', choices = ["sum-over", "max-pool", "mean-pool"])
parser.add_argument('-lin', help='use lin-tasc', action='store_true')
parser.add_argument('-feat', help='use feat-tasc', action='store_true')
parser.add_argument('-conv', help='use conv-tasc', action='store_true')

print("\n", vars(parser.parse_args()), "\n")

args = vars(parser.parse_args())

dataset = args["dataset"]
encode_select = args["encoder"]
data_dir= args["data_dir"]
model_dir = args["model_dir"] 
sys.path.append(data_dir)
method = [k for k,v in args.items() if v is True]
save_path = [model_dir + method[0] + "_" + dataset + "/" if len(method) > 0 else model_dir + dataset + "/"][0]

experiments_path = [args["experiments_dir"] + method[0] + "_" + dataset + "/" if len(method) > 0 else args["experiments_dir"] + dataset + "/"][0]

try:

  os.makedirs(save_path + "/model_run_stats/")
  print("\n--Models saved in: {}".format(save_path))

except:

  print("\n--Models saved in: {}".format(save_path))
  
try:

  os.makedirs(experiments_path)
  print("--Experiment results saved in {}".format(experiments_path))

except:

  print("--Experiment results saved in {}".format(experiments_path))

args["bert_model"] = "bert-base-uncased"

if args["dataset"] == "mimicanemia":

  args["bert_model"] = "bert-base-uncased"#"allenai/scibert_scivocab_uncased"

seeds = [24,92, 7]

if args["encoder"] == "bert":

    data = dataholder_bert(data_dir, dataset, 8, args["bert_model"])
    
else:

    
    data = dataholder(data_dir, dataset, 32)

# In[18]:


vocab_size = data.vocab_size
embedding_dim = data.embedding_dim
hidden_dim = data.hidden_dim


tasc_method = method[0] if len(method) > 0 else None
hidden_dim = 64 if args["encoder"] != "bert" else 768 // 2
embedding_dim = 300 if args["encoder"] != "bert" else 768
epochs = 20 if args["encoder"] != "bert" else 10 

## special case for twitter without tasc and tanh (not balaned dataset)
if (args["dataset"] == "twitter") & (tasc_method == None):

    epochs = 30

    if args["encoder"] == "mlp":

        epochs = 35


args.update({"vocab_size" : vocab_size, "embedding_dim" : embedding_dim, "hidden_dim" : hidden_dim, 
             "tasc" : tasc_method , "seeds" : seeds, "optimiser":"Adam", "loss":"CrossEntropyLoss",
             "epochs" : epochs, "save_path":save_path, "experiments_path": experiments_path})

print(args)

#### saving config file for this run

with open('modules/config.txt', 'w') as file:
     file.write(json.dumps(args))
     

### training and evaluating models

from modules.run_binary_classification import *

print("\nTraining\n")

train_binary_classification_model(data)
     
print("\Evaluating\n")

evaluate_trained_bc_model(data)


### conducting experiments

## special case for mimic

if args["encoder"] == "bert":

    if (args["dataset"] == "mimicanemia" and args["tasc"] is None):

        del data

        data = dataholder_bert(data_dir, dataset, 4, args["bert_model"])

    if (args["dataset"] == "imdb" and args["tasc"] is None):

        del data

        data = dataholder_bert(data_dir, dataset, 4, args["bert_model"])
    
    if (args["dataset"] == "mimicanemia" and args["tasc"] is not None):

        del data

        data = dataholder_bert(data_dir, dataset, 2, args["bert_model"])
    
    if (args["dataset"] == "imdb" and args["tasc"] is not None):

        del data

        data = dataholder_bert(data_dir, dataset, 2, args["bert_model"])


print("\nExperiments\n")

conduct_experiments(data)
#



