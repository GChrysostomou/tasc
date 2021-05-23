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
from captum.attr import IntegratedGradients

"""MODEL OUT DISTRIBUTIONS"""

with open('modules/config.txt', 'r') as f:
    args = json.load(f)

def percentage_removed(data, model, save_path):
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(100)
    np.random.seed(100)
    
    pbar = trange(len(data) * data.batch_size, desc='set of tokens', leave=True)

    results_flip = {}
    results_flip["max_source"] = []
    results_flip["random"] = []
    results_flip["lengths"] = []
    results_flip["att_grad"] = []
    results_flip["att*grad"] = []

    if args["tasc"] is None:
  
        results_flip["grad"] = []
        results_flip["omission"] = []
        results_flip["IG"] = []
    
    
    if args["encoder"] == "bert":
        model.encoder.bert.embeddings.requires_grad_(True)

    else:
        model.encoder.embedding.weight.requires_grad_(True)


    for sentences, lengths, labels in data:
        
        torch.cuda.empty_cache()
        model.zero_grad()

        if args["encoder"] == "bert":
            
            sentences, lengths, labels = torch.stack(sentences).transpose(0,1).to(device), lengths.to(device), labels.to(device)
                
        else:
            
            sentences, lengths, labels = Variable(sentences).to(device),Variable(lengths).to(device), Variable(labels).to(device)

        # original trained model    
        model.train()
        
    
        yhat, weights_or = model(sentences, lengths, retain_gradient = True)
        
        yhat.max(-1)[0].sum().backward(retain_graph = True)

        if args["tasc"] is None:

            g = model.encoder.embed.grad

            em = model.encoder.embed
    
            g1 = (g* em).sum(-1)[:,:max(lengths)]

            integrated_grads = model.integrated_grads(sentences, 
                                                    g, lengths, 
                                                    original_pred = yhat.max(-1))
         

        weights_def_grad = model.weights.grad

        with torch.no_grad():

            att_source_set = {}
            rand_set = {}
            att_grad_set = {}
            att_mul_grad_set = {}

            if args["tasc"] is None:

                grad_set = {}
                ig_set = {}
                omission_set = {}
                
                g1.masked_fill_(model.masks[:,:max(lengths)].bool(), float("-inf"))
                top_grad = torch.topk(g1, k = g1.size(1))[1].to(device)


                omission_scores = model.get_omission_scores(sentences, lengths, yhat)
                top_omission = torch.topk(omission_scores, k = weights_or.size(1))[1].to(device)

                integrated_grads.masked_fill_(model.masks[:,:max(lengths)].bool(), float("-inf"))
                top_IG = torch.topk(integrated_grads, k = integrated_grads.size(1))[1].to(device)

            top_att = torch.topk(weights_or, k = weights_or.size(1))[1].to(device)
            top_randi = torch.randn(weights_or.shape)
            top_rand = torch.topk(top_randi, k = weights_or.size(1))[1].to(device)
        
            weight_mul_grad = weights_or * weights_def_grad
            
            weights_def_grad.masked_fill_(model.masks[:,:max(lengths)].bool(), float("-inf"))
            top_att_grad = torch.topk(weights_def_grad, k = weights_or.size(1))[1].to(device)
            
            
            weight_mul_grad.masked_fill_(model.masks[:,:max(lengths)].bool(),float("-inf"))
            top_att_mul_grad = torch.topk(weight_mul_grad, k = weights_or.size(1))[1].to(device)
            

            temp = 0
            
            model.eval()
           
            maximum = max(lengths)
            
            lengths_ref = lengths.clone()
            
            rows = torch.arange(sentences.size(0)).long().to(device)

            for _j_ in range(0,maximum):
                
                """Attention at source"""
                
                mask = torch.zeros(sentences.shape).to(device)
               
                mask = mask.scatter_(1,  top_att[rows, _j_+1:], 1)
            
                yhat_max_source, _ = model(sentences.float() * mask.float(), lengths)
                
                check_indexes_max_s = (yhat.max(-1)[1] != yhat_max_source.max(-1)[1]).nonzero()
                    
                if check_indexes_max_s.nelement() != 0:
                    for atin in check_indexes_max_s:
           
                        
                        if atin.item() not in att_source_set.keys():
                            
                            temp += 1
                            att_source_set[atin.item()] = (_j_ + 1) / lengths_ref[atin].item() 
                            
                        else:
                            
                            pass
                        
                """Attention gradient at source"""
                
                mask = torch.zeros(sentences.shape).to(device)
               
                mask = mask.scatter_(1,  top_att_grad[rows, _j_+1:], 1)
            
                yhat_grad_att_source, _ = model(sentences.float() * mask.float(), lengths)
                
                check_indexes_grad_att_s = (yhat.max(-1)[1] != yhat_grad_att_source.max(-1)[1]).nonzero()
                    
                if check_indexes_grad_att_s.nelement() != 0:
                    for atin in check_indexes_grad_att_s:
           
                        
                        if atin.item() not in att_grad_set.keys():
                            
                            temp += 1
                            att_grad_set[atin.item()] = (_j_ + 1) / lengths_ref[atin].item() 
                            
                        else:
                            
                            pass
                        
                """Attention * gradient at source"""
                
                mask = torch.zeros(sentences.shape).to(device)
               
                mask = mask.scatter_(1,  top_att_mul_grad[rows, _j_+1:], 1)
                
                yhat_grad_mul_att_source, _ = model(sentences.float() * mask.float(), lengths)
                
                check_indexes_grad_mul_att_s = (yhat.max(-1)[1] != yhat_grad_mul_att_source.max(-1)[1]).nonzero()
                    
                if check_indexes_grad_mul_att_s.nelement() != 0:
                    for atin in check_indexes_grad_mul_att_s:
           
                        if atin.item() not in att_mul_grad_set.keys():
                            
                            temp += 1
                            att_mul_grad_set[atin.item()] = (_j_ + 1) / lengths_ref[atin].item() 
                            
                        else:
                            
                            pass

                if args["tasc"] is None:

                    """Gradient"""
                    
                    mask = torch.zeros(sentences.shape).to(device)
                
                    mask = mask.scatter_(1,  top_grad[rows, _j_+1:], 1)
                
                    yhat_grad,_ = model(sentences.float() * mask.float(), lengths)
        
                    check_indexes_grad = (yhat.max(-1)[1] != yhat_grad.max(-1)[1]).nonzero()
                    
                    if check_indexes_grad.nelement() != 0:
                        
                        for items in check_indexes_grad:
                                        
                            if items.item() not in grad_set.keys():
                        
                                temp += 1
                                grad_set[items.item()] = (_j_ + 1) / lengths_ref[items].item()
                                
                            else:
                                
                                pass


                    """INTEGRATED Gradient"""
                    
                    mask = torch.zeros(sentences.shape).to(device)
                
                    mask = mask.scatter_(1,  top_IG[rows, _j_+1:], 1)
                
                    yhat_IG,_ = model(sentences.float() * mask.float(), lengths)
        
                    check_indexes_IG = (yhat.max(-1)[1] != yhat_IG.max(-1)[1]).nonzero()
                    
                    if check_indexes_IG.nelement() != 0:
                        
                        for items_IG in check_indexes_IG:
                                        
                            if items_IG.item() not in ig_set.keys():
                        
                                temp += 1
                                ig_set[items_IG.item()] = (_j_ + 1) / lengths_ref[items_IG].item()
                                
                            else:
                                
                                pass
        


                    """Ommision"""
                    
                    mask = torch.zeros(sentences.shape).to(device)
                
                    mask = mask.scatter_(1,  top_omission[rows, _j_+1:], 1)
                    
                    yhat_omission, _ = model(sentences.float() * mask.float(), lengths)
                    
                    check_indexes_omission = (yhat.max(-1)[1] != yhat_omission.max(-1)[1]).nonzero()
                        
                    if check_indexes_omission.nelement() != 0:
                        for omi in check_indexes_omission:
            
                            if omi.item() not in omission_set.keys():
                                
                                temp += 1
                                omission_set[omi.item()] = (_j_ + 1) / lengths_ref[omi].item() 
                                
                            else:
                                
                                pass
                
    
                """Random"""
                
                mask = torch.zeros(sentences.shape).to(device)
               
                mask = mask.scatter_(1,  top_rand[rows, _j_+1:], 1)
                
                
                yhat_rand, _ = model(sentences.float() * mask.float(), lengths)
                
                check_indexes_rand  = (yhat.max(-1)[1] != yhat_rand.max(-1)[1]).nonzero()
                
                if check_indexes_rand.nelement() != 0:
                    
                    for rna in check_indexes_rand:
                                            
                        if rna.item() not in rand_set.keys():
                     
                            rand_set[rna.item()] = (_j_ + 1) / lengths_ref[rna].item()
                            
                        else:
                            
                            pass
                        

            
            for _i_ in range(0, sentences.size(0)):
                
                if _i_ not in rand_set.keys():
                    
                    rand_set[_i_] = 1
       
                if _i_ not in att_source_set.keys():
                    
                    att_source_set[_i_] = 1
            
                if _i_ not in att_grad_set.keys():
                    
                    att_grad_set[_i_] = 1
                    
                if _i_ not in att_mul_grad_set.keys():
                    
                    att_mul_grad_set[_i_] = 1
                
                if args["tasc"] is None:

                    if _i_ not in omission_set.keys():
                        
                        omission_set[_i_] = 1

                        
                    if _i_ not in grad_set.keys():
                        
                        grad_set[_i_] = 1

                    if _i_ not in ig_set.keys():
                        
                        ig_set[_i_] = 1

                    

            
            att_mul_grad_set = {k:(1 if v > 1 else v) for k,v in att_mul_grad_set.items()}
            att_grad_set = {k:(1 if v > 1 else v) for k,v in att_grad_set.items()}
            rand_set = {k:(1 if v > 1 else v) for k,v in rand_set.items()}
            att_source_set = {k:(1 if v > 1 else v) for k,v in att_source_set.items()}
            

            att_mul_grad_set = OrderedDict(sorted(att_mul_grad_set.items()))
            att_grad_set = OrderedDict(sorted(att_grad_set.items()))
            rand_set = OrderedDict(sorted(rand_set.items()))
            att_source_set = OrderedDict(sorted(att_source_set.items()))

            if args["tasc"] is None:
                
                ig_set = {k:(1 if v > 1 else v) for k,v in ig_set.items()}
                grad_set = {k:(1 if v > 1 else v) for k,v in grad_set.items()}
                omission_set = {k:(1 if v > 1 else v) for k,v in omission_set.items()}
                
                ig_set = OrderedDict(sorted(ig_set.items()))
                grad_set = OrderedDict(sorted(grad_set.items()))
                omission_set = OrderedDict(sorted(omission_set.items()))

            if len(yhat.shape) == 1:
                
                pass
                
            else:
                
                
                results_flip["random"].extend(rand_set.values())
                results_flip["max_source"].extend(att_source_set.values())
                results_flip["att_grad"].extend(att_grad_set.values())
                results_flip["lengths"].extend(lengths.cpu().data.numpy())
                results_flip["att*grad"].extend(att_mul_grad_set.values())

                if args["tasc"] is None:

                    results_flip["grad"].extend(grad_set.values())
                    results_flip["omission"].extend(omission_set.values())
                    results_flip["IG"].extend(ig_set.values())
    
        pbar.update(data.batch_size)
        pbar.refresh()

    
    """Saving percentage decision flip"""
 
    
    df = pd.DataFrame.from_dict(results_flip)
    

    df.to_csv(save_path + "_decision-flip-set.csv")
    
    
    df = df.drop(columns = "lengths")

    summary = df.mean(axis = 0)

    summary.to_csv(save_path + "_decision-flip-set-summary.csv", header = ["mean percentage"])
