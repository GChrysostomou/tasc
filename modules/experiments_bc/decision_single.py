import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pandas as pd
from modules.experiments_bc.eval_utils import * 
from tqdm import trange
from modules.experiments_bc.decision_set import register_flips_

import json

with open('modules/config.txt', 'r') as f:
    args = json.load(f)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def effect_on_output(data, model, save_path):
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(100)
    np.random.seed(100)
    
    pbar = trange(
        len(data) * data.batch_size, 
        desc=f"running experiments for fraction of tokens on test", 
        leave=True
    )
    

    flip_results = {}
    
    if args["encoder"] == "bert":
        model.encoder.bert.embeddings.requires_grad_(True)

    else:
        model.encoder.embedding.weight.requires_grad_(True)

    counter = 0
    
    for sentences, lengths, labels in data:
        
        torch.cuda.empty_cache()
        model.zero_grad()
        model.eval()

        if args["encoder"] == "bert":
            
            sentences, lengths, labels = torch.stack(sentences).transpose(0,1).to(device), lengths.to(device), labels.to(device)
                
        else:
            
            sentences, lengths, labels = Variable(sentences).to(device),Variable(lengths).to(device), Variable(labels).to(device)

        index_list = []

        for _i_ in range(sentences.size(0)):
            
            index = f"test-{counter}"

            flip_results.update({index:{}})
            
            counter+=1
            index_list.append(index)

        # original trained model    
        model.train()
        
    
        yhat, weights_or = model(sentences, lengths, retain_gradient = True)

        yhat.max(-1)[0].sum().backward(retain_graph = True)

        g = model.encoder.embed.grad

        em = model.encoder.embed

        g1 = (g* em).sum(-1)[:,:max(lengths)]

        weights_def_grad = model.weights.grad
        random = torch.randn(weights_or.shape)

        g1.masked_fill_(model.masks[:,:max(lengths)].bool(), float("-inf"))

        weight_mul_grad = weights_or * weights_def_grad
        
        weights_def_grad.masked_fill_(model.masks[:,:max(lengths)].bool(), float("-inf"))      
        
        weight_mul_grad.masked_fill_(model.masks[:,:max(lengths)].bool(),float("-inf"))

        maximum = max(lengths)
        increments =  torch.round(maximum.float() * 0.02).int()
        increments = max(1,increments)
                
        maximum = max(lengths)
        
        lengths_ref = lengths.clone()
        
        rows = torch.arange(sentences.size(0)).long().to(device)

        original_sentences = sentences.clone().detach()

        model.eval()
        with torch.no_grad():
            
            for feat_name, feat_score in {"random" : random, "attention" : weights_or, \
                                        "gradients" : g1, "scaled attention" : weight_mul_grad, \
                                        "attention gradients" : weights_def_grad}.items():

                feat_rank = torch.topk(feat_score, k = feat_score.size(1))[1].to(device)

                register_flips_(
                    model = model, 
                    ranking = feat_rank, 
                    original_prediction = yhat, 
                    original_sentences = original_sentences, 
                    rows = rows, 
                    results_dictionary = flip_results, 
                    no_of_tokens = 0, 
                    feat_attr_name = feat_name,
                    lengths = lengths_ref,
                    indexes= index_list,
                    binary = True
                )


        pbar.update(data.batch_size)
        pbar.refresh()

    ### if we didnt register any flips for particular instances
    ## it means we reached the max so fraction of is 1.
    for annot_id in flip_results.keys():

        for feat_name in {"random", "attention", "gradients", "scaled attention", "attention gradients"}:

            if feat_name not in flip_results[annot_id]:

                flip_results[annot_id][feat_name] = False

    """Saving percentage decision flip"""
    
    df = pd.DataFrame.from_dict(flip_results).T
    
    df.to_csv(save_path + "_decision-flip-single.csv")
    
    summary = df.mean(axis = 0) * 100

    summary.to_csv(save_path + "_decision-flip-single-summary.csv", header = ["mean percentage"])
    
    return

