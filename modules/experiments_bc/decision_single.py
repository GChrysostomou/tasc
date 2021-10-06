import torch
import matplotlib
matplotlib.use("Agg")
from torch.autograd import Variable
import pandas as pd
from modules.experiments_bc.eval_utils import * 
from tqdm import trange
from modules.experiments_bc.decision_set import register_flips_
import os
import json
from modules.utils import batch_from_dict_

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
        desc=f"running experiments for fraction of tokens (SINGLE TOKEN) on test", 
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
    fname = f"{save_path}_decision-flip-single-summary.json"

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

        lengths_ref = lengths.clone()
        
        rows = torch.arange(sentences.size(0)).long().to(device)

        original_sentences = sentences.clone().detach()

        with torch.no_grad():
            
            for feat_name in {"random" , "attention" , "gradients" , "scaled attention", 
                                        "attention gradients" }:

                feat_score =  batch_from_dict_(
                    inst_indx = inst_idx, 
                    metadata = importance_scores, 
                    target_key = feat_name,
                )

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
                    indexes= inst_idx,
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
    df["instance_idx"] = df.index

    with open(f"{save_path}_decision-flip-single.json", "w")as file:
        
        json.dump(
            df.to_dict("records"),
            file,
            indent = 4
        )
    
    summary = df.mean(axis = 0) * 100

    with open(f"{save_path}_decision-flip-single-summary.json", "w")as file:
        
        json.dump(
            summary.to_dict(),
            file,
            indent = 4
        )

    return

