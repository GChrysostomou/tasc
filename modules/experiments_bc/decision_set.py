import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pandas as pd
from modules.experiments_bc.eval_utils import * 
from collections import OrderedDict
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from tqdm import trange
import json

with open('modules/config.txt', 'r') as f:
    args = json.load(f)

def register_flips_(model , ranking , original_prediction , original_sentences, 
    rows , results_dictionary , no_of_tokens , 
    feat_attr_name , lengths , indexes , binary = False
    ):

    """
    registers if flips occur in an immutable dictionary
    Inputs: 
        model : torch.nn.Module -> finetuned model
        model_inputs : dic -> dictionary containing the model inputs
        ranking : torch.tensor -> rankings of words from most important to least
        original_prediction : torch.tensor -> model output predictions (distribution)
        original_sentences : torch.tensor -> original input ids
        rows : torch.arange -> just used to select in the ranking (size batch)
        results_dictionary : dic -> nested dictionary to store the results in the format dic[instance_idx][feature_attribution]
        no_of_tokens : int -> number of tokens to be masked
        feat_attr_name : str -> name of feature attribution used for storing results
    Outputs:
        returns None -> results stored in results_dictionary    
    """

    mask = torch.zeros(original_sentences.shape).to(device)
               
    mask = mask.scatter_(1,  ranking[rows, no_of_tokens+1:], 1)

    sentences = (original_sentences.float() * mask.float()).long()

    masked_prediction, _ = model(sentences.float() * mask.float(), lengths)
    
    flips = (original_prediction.max(-1)[1] != masked_prediction.max(-1)[1]).nonzero()

    if flips.nelement() != 0: 

        for indx in flips:

            annotation_id = indexes[indx]
            
            if feat_attr_name not in results_dictionary[annotation_id].keys():
                
                ## for single flip results:
                if binary:
                    
                    results_dictionary[annotation_id][feat_attr_name] = True

                else:

                    results_dictionary[annotation_id][feat_attr_name] = (no_of_tokens + 1) / lengths[indx].item() 
                


            else:

                pass

    return



def percentage_removed(data, model, save_path):
    
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

        integrated_grads = model.integrated_grads(
            sentences, 
            g, 
            lengths, 
            original_pred = yhat.max(-1)
        )
         

        weights_def_grad = model.weights.grad
        random = torch.randn(weights_or.shape)

        g1.masked_fill_(model.masks[:,:max(lengths)].bool(), float("-inf"))


        omission_scores = model.get_omission_scores(
            sentences, 
            lengths, 
            yhat
        )

        integrated_grads.masked_fill_(model.masks[:,:max(lengths)].bool(), float("-inf"))

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
                                        "gradients" : g1, "scaled attention" : weight_mul_grad, "ig": integrated_grads, \
                                        "attention gradients" : weights_def_grad, "omission": omission_scores}.items():

                feat_rank = torch.topk(feat_score, k = feat_score.size(1))[1].to(device)

                for no_of_tokens in range(0,maximum+increments, increments):

                    register_flips_(
                        model = model, 
                        ranking = feat_rank, 
                        original_prediction = yhat, 
                        original_sentences = original_sentences, 
                        rows = rows, 
                        results_dictionary = flip_results, 
                        no_of_tokens = no_of_tokens, 
                        feat_attr_name = feat_name,
                        lengths = lengths_ref,
                        indexes= index_list
                    )


        pbar.update(data.batch_size)
        pbar.refresh()


    ### if we didnt register any flips for particular instances
    ## it means we reached the max so fraction of is 1.
    for annot_id in flip_results.keys():

        for feat_name in {"random", "attention", "gradients", "scaled attention", "ig", "attention gradients", "omission"}:

            if feat_name not in flip_results[annot_id]:

                flip_results[annot_id][feat_name] = 1.

    """Saving percentage decision flip"""
    
    df = pd.DataFrame.from_dict(flip_results).T
    

    df.to_csv(save_path + "_decision-flip-set.csv")
    
    summary = df.mean(axis = 0)

    summary.to_csv(save_path + "_decision-flip-set-summary.csv", header = ["mean percentage"])

    return