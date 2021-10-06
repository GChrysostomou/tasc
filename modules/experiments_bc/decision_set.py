import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pandas as pd
from modules.experiments_bc.eval_utils import * 
from modules.utils import batch_from_dict_
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

import os

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

    ## import importance_scores
    fname = os.path.join(
        os.getcwd(), 
        "importance_scores", 
        save_path.split('/')[-2],
        ""
    )

    if os.path.exists(fname + f"{save_path.split('/')[-1]}-importance_scores.npy"):

        print(f"** Importance scores already extracted in -->> {fname}")
        
        importance_scores = np.load(
            fname + f"{save_path.split('/')[-1]}-importance_scores.npy",
            allow_pickle = True
        ).item()

    else:

        raise FileNotFoundError(
            """
            Please run the function to retrieve importance scores
            """
        )

    ## check if descriptors already exist
    fname = f"{save_path}_decision-flip-set-summary.json"

    if os.path.exists(fname):

        print("**** results already exist. Delete if you want a rerun")
        return

        
    for inst_idx, sentences, lengths, labels in data:
        
        torch.cuda.empty_cache()
        model.zero_grad()
        model.eval()

        if args["encoder"] == "bert":
            
            sentences, lengths, labels = torch.stack(sentences).transpose(0,1).to(device), lengths.to(device), labels.to(device)
                
        else:
            
            sentences, lengths, labels = Variable(sentences).to(device),Variable(lengths).to(device), Variable(labels).to(device)

        for indx in inst_idx:
            
            flip_results.update({indx:{}})
    
        yhat, _ = model(sentences, lengths, retain_gradient = True)

                
        maximum = max(lengths)
        
        lengths_ref = lengths.clone()
        
        rows = torch.arange(sentences.size(0)).long().to(device)

        original_sentences = sentences.clone().detach()

        model.eval()

        if args["speed_up"]:

            increments =  torch.round(maximum.float() * 0.02).int()
            increments = max(1,increments)

            mirange = range(0,maximum+increments, increments)

        else:

            mirange = range(0, maximum)

        with torch.no_grad():
            
            for feat_name in {"random" , "attention","gradients" , "scaled attention", "ig",
                            "attention gradients" }:

                feat_score =  batch_from_dict_(
                    inst_indx = inst_idx, 
                    metadata = importance_scores, 
                    target_key = feat_name,
                )


                feat_rank = torch.topk(feat_score, k = feat_score.size(1))[1].to(device)

                for no_of_tokens in mirange:

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
                        indexes= inst_idx
                    )


        pbar.update(data.batch_size)
        pbar.refresh()


    ### if we didnt register any flips for particular instances
    ## it means we reached the max so fraction of is 1.
    for annot_id in flip_results.keys():

        for feat_name in {"random", "attention", "gradients", "scaled attention", "ig", "attention gradients"}:

            if feat_name not in flip_results[annot_id]:

                flip_results[annot_id][feat_name] = 1.

    """Saving percentage decision flip"""
    
    df = pd.DataFrame.from_dict(flip_results).T

    df["instance_idx"] = df.index

    with open(f"{save_path}_decision-flip-set.json", "w")as file:
        
        json.dump(
            df.to_dict("records"),
            file,
            indent = 4
        )
        
    summary = df.mean(axis = 0)

    with open(f"{save_path}_decision-flip-set-summary.json", "w")as file:
        
        json.dump(
            summary.to_dict(),
            file,
            indent = 4
        )

    return