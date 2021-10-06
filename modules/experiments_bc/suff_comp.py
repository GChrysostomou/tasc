import torch
import matplotlib
matplotlib.use("Agg")
from torch.autograd import Variable
from modules.experiments_bc.eval_utils import * 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from tqdm import trange
import json
from modules.utils import normalized_comprehensiveness_, normalized_sufficiency_, sufficiency_, create_rationale_mask_, batch_from_dict_
import gc
import os

with open('modules/config.txt', 'r') as f:
    args = json.load(f)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def conduct_tests_(data, model, save_path):

    pbar = trange(len(data) * data.batch_size, desc="running for AOPC", leave=True)
    
    faithfulness_results = {}

    counter = 0

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
    fname = save_path + "-faithfulness-scores-averages--description.json"

    if os.path.exists(fname):

        print("**** results already exist. Delete if you want a rerun")
        return

    for inst_idx, sentences, lengths, labels in data:
        
        torch.cuda.empty_cache()
        model.zero_grad()

        if args["encoder"] == "bert":
            
            sentences, lengths, labels = torch.stack(sentences).transpose(0,1).to(device), lengths.to(device), labels.to(device)
                
        else:
            
            sentences, lengths, labels = Variable(sentences).to(device),Variable(lengths).to(device), Variable(labels).to(device)

        index_list = []

        for _i_ in range(sentences.size(0)):
            
            index = f"test-{counter}"

            faithfulness_results.update({index:{}})
            
            counter+=1
            index_list.append(index)

        original_prediction, _ = model(
            sentences, 
            lengths, 
            retain_gradient = True
        )

        original_sentences = sentences.clone().detach()[:,:max(lengths)]

        model.eval()

        original_prediction = torch.softmax(original_prediction, dim = -1).cpu().detach().numpy()
        full_text_class = original_prediction.argmax(-1)
        full_text_probs = original_prediction.max(-1)

        yhat, _  = model(
            original_sentences, 
            lengths, 
            retain_gradient = False,  
            ig = 1e-16 
        )

        yhat = torch.softmax(yhat, dim = -1).cpu().detach().numpy()

        rows = torch.arange(sentences.size(0)).long().cpu().numpy()

        if len(rows) == 1:

            rows = rows[0]
            reduced_probs = [yhat[full_text_class]]

        else:

            reduced_probs = yhat[rows, full_text_class]

        ## baseline sufficiency
        suff_y_zero = sufficiency_(
            full_text_probs, 
            reduced_probs
        )

        ## AOPC scores and other metrics
        rationale_ratios = [0.02, 0.1, 0.2, 0.5]

        for rationale_type in {"topk"}:

            for _j_, annot_id in enumerate(index_list):
                    
                faithfulness_results[annot_id]["full text prediction"] = original_prediction[_j_] 
                faithfulness_results[annot_id]["true label"] = labels[_j_].detach().cpu().item()
            
            for feat_name in {"random" , "attention", "gradients" , "ig" , 
                             "scaled attention" , "attention gradients"}:

                feat_score =  batch_from_dict_(
                    inst_indx = inst_idx, 
                    metadata = importance_scores, 
                    target_key = feat_name,
                )

                suff_aopc = np.zeros([yhat.shape[0], len(rationale_ratios)], dtype=np.float64)
                comp_aopc = np.zeros([yhat.shape[0], len(rationale_ratios)], dtype=np.float64)

                for _i_, rationale_length in enumerate(rationale_ratios):
                    
                    ## if we are masking for a query that means we are preserving
                    ## the query and we DO NOT mask it
                  
                    rationale_mask = create_rationale_mask_(
                        importance_scores = feat_score, 
                        no_of_masked_tokens = torch.ceil(lengths.float() * rationale_length).cpu().detach().cpu().numpy(),
                        method = rationale_type
                    )

                    ## measuring faithfulness
                    comp, _  = normalized_comprehensiveness_(
                        model = model, 
                        original_sentences = original_sentences, 
                        rationale_mask = rationale_mask, 
                        inputs = {"input" : original_sentences, "lengths":lengths}, 
                        full_text_probs = full_text_probs, 
                        full_text_class = full_text_class, 
                        rows = rows,
                        suff_y_zero = suff_y_zero
                    )

                    suff, _ = normalized_sufficiency_(
                        model = model, 
                        original_sentences = original_sentences, 
                        rationale_mask = rationale_mask, 
                        inputs = {"input" : original_sentences, "lengths":lengths}, 
                        full_text_probs = full_text_probs, 
                        full_text_class = full_text_class, 
                        rows = rows,
                        suff_y_zero = suff_y_zero
                    )

                    suff_aopc[:,_i_] = suff
                    comp_aopc[:,_i_] = comp

                for _j_, annot_id in enumerate(index_list):
                    
                    faithfulness_results[annot_id][feat_name] = {
                        "sufficiency aopc" : {
                            "mean" : suff_aopc[_j_].sum() / (len(rationale_ratios) + 1),
                            "per ratio" : suff_aopc[_j_]
                        },
                        "comprehensiveness aopc" : {
                            "mean" : comp_aopc[_j_].sum() / (len(rationale_ratios) + 1),
                            "per ratio" : comp_aopc[_j_]
                        }
                    }
        
                del feat_score
                del rationale_mask
                del suff
                del comp

                gc.collect()
                torch.cuda.empty_cache()

        del sentences
        del original_sentences
        del lengths
        del yhat

        gc.collect()

        torch.cuda.empty_cache()

        pbar.update(data.batch_size)

            
    descriptor = {}
    # filling getting averages
    for feat_attr in {"attention", "gradients", "ig", "random", "scaled attention", "attention gradients"}:
        
        aopc_suff= np.asarray([faithfulness_results[k][feat_attr][f"sufficiency aopc"]["mean"] for k in faithfulness_results.keys()])
        aopc_comp = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness aopc"]["mean"] for k in faithfulness_results.keys()])

        descriptor[feat_attr] = {
            "AOPC - sufficiency" : {
                "mean" : aopc_suff.mean(),
                "std" : aopc_suff.std()
            },
            "AOPC - comprehensiveness" : {
                "mean" : aopc_comp.mean(),
                "std" : aopc_comp.std()
            }
        }

    ## save all info
    fname = save_path + "-faithfulness-scores-detailed-.npy"

    np.save(fname, faithfulness_results)

    ## save descriptors
    fname = save_path + "-faithfulness-scores-averages--description.json"

    with open(fname, 'w') as file:
            json.dump(
                descriptor,
                file,
                indent = 4
            ) 

    return