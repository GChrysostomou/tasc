import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import *
from sklearn.metrics import precision_recall_fscore_support as prfs

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def degrading_model_perf(data, model, save_path, data_size, largest = True):
  
    print("\n--- Degrading Model Performance \n")
    
    modulo = round(len(data) / 10) + 1
    
    model.embedding.weight.requires_grad_(True)
    
    actual = []    
    
    results = {}
    results["random"] = []
    results["attention"]= []
    results["gradient"] = []
    results["grad_attention"] = []
    results["grad*attention"] = []
    
    _, _, lengths, _  = next(iter(data))
    
    
    maximum = max(lengths)
     
    if max(lengths) <= 10 :
        
        maximum = max(lengths) - 1
        
    elif max(lengths) > 10 :
        
        maximum = 10
    
    print(maximum)
    
    
    grad_set = torch.zeros([data_size, maximum]).long().to(device)
    att_set = torch.zeros([data_size, maximum]).long().to(device)
    rand_set = torch.zeros([data_size, maximum]).long().to(device)
    att_grad_set = torch.zeros([data_size, maximum]).long().to(device)
    att_x_grad_set = torch.zeros([data_size, maximum]).long().to(device)
    actual_set = torch.zeros([data_size, 1]).long().to(device)
    docs = []
    
    for batchi, (doc_id, sentences, lengths, labels) in enumerate(data):
        model.train()
        torch.cuda.empty_cache()
        model.zero_grad()
        
        sentences, lengths, labels = sentences.to(device), lengths.to(device), labels.to(device)

        yhat, weights_or = model(sentences, lengths, retain_gradient = True)
     
        
        masking = yhat.max(-1)[1] == labels
        
        if largest == False:
            
            masking = yhat.max(-1)[1] != labels
            
        yhat.max(-1)[0].sum().backward(retain_graph = True)
  
        maxi = max(lengths)
        
        doc_id = doc_id[masking]
        yhat = yhat[masking]
        
        sentences = sentences[masking]
        labels = labels[masking]
        lengths = lengths[masking]
        weights_or = weights_or[masking]
        
        
        docs.extend(doc_id)
        g = model.embed.grad[masking]
        weights_def_grad = model.weights.grad[masking]
        
        max_lengths = max(max(lengths), maxi)
        
        model_masks = model.masks[masking]
        
        with torch.no_grad():
            
            
            weights = weights_or.clone()
            
            weight_mul_grad = weights_or * weights_def_grad
            weight_mul_grad[model_masks[:,:max_lengths]] = float("-inf")
           
            weights_def_grad_soft = weights_def_grad.clone()
            weights_def_grad_soft[model_masks[:,:max_lengths]] = float("-inf")
            
            em = model.embed[masking]
    
            g1 = (g* em).sum(-1)[:,:max_lengths]
            g1[model_masks[:,:max_lengths]] = float("-inf")
            
            sentence_att = sentences.clone()[:,:max_lengths]
            sentence_grad = sentences.clone()[:,:max_lengths]
            sentence_rand = sentences.clone()[:,:max_lengths]
            sentence_att_grad = sentences.clone()[:,:max_lengths]
            sentence_att_mul_grad = sentences.clone()[:,:max_lengths]
      
            g1[model_masks[:,:max_lengths]] = float("-inf")
            
            top_grad = torch.topk(g1, k = g1.size(1), largest = largest)[1]
            
            top_att = torch.topk(weights, k = weights.size(1), 
                                 largest = largest)[1]
           
            
            top_rand = torch.randn(top_att.shape)
          
            top_rand = torch.topk(top_rand, k = weights.size(1), 
                                  largest = largest)[1]
    
            top_att_grad = torch.topk(weights_def_grad_soft, 
                                      k = weights.size(1), 
                                      largest = largest)[1]
            
            top_att_mul_grad = torch.topk(weight_mul_grad, 
                                          k = weights.size(1), 
                                          largest = largest)[1]
             
            temp_pred = []
            temp_act = []
            
            temp_act.append(labels.cpu().data.numpy())
            temp_pred.append(yhat.max(-1)[1].cpu().data.numpy())
            model.eval()
                      
            actual_set[doc_id] = labels.unsqueeze(-1)
            
            rand_set[doc_id, 0] = yhat.max(-1)[1] 
            att_set[doc_id, 0] = yhat.max(-1)[1] 
            grad_set[doc_id, 0] = yhat.max(-1)[1] 
            att_grad_set[doc_id, 0] = yhat.max(-1)[1] 
            att_x_grad_set[doc_id, 0] = yhat.max(-1)[1] 
            
            rows = torch.arange(sentences.size(0))
            

            for _j_ in range(1,maximum):
                
                sentence_grad[rows, top_grad[:,_j_]] = 0
                           
                sentence_att[rows, top_att[:,_j_]] = 0
                
                sentence_att_grad[rows, top_att_grad[:,_j_]] = 0
                
                sentence_att_mul_grad[rows, top_att_mul_grad[:,_j_]] = 0
                
                sentence_rand[rows, top_rand[:,_j_]] = 0
                
                yhat_rand, _ = model(sentence_rand,lengths)
           
                rand_set[doc_id, _j_] = yhat_rand.max(-1)[1]   
                
                yhat_att, _ = model(sentence_att,lengths)
           
                att_set[doc_id, _j_] = yhat_att.max(-1)[1]  
                
                yhat_grad, _ = model(sentence_grad,lengths)
           
                grad_set[doc_id, _j_] = yhat_grad.max(-1)[1] 
                
                yhat_att_grad, _ = model(sentence_att_grad,lengths)
                
                att_grad_set[doc_id, _j_] = yhat_att_grad.max(-1)[1] 
                
                yhat_att_x_grad, _ = model(sentence_att_mul_grad,lengths)
                
                att_x_grad_set[doc_id, _j_] = yhat_att_x_grad.max(-1)[1] 
          
        if batchi % modulo == 0 :
            
            print("Remaining: ", len(data)- batchi)


    

    docs = torch.LongTensor(docs)
    
    rand_set = rand_set[docs]
    att_set = att_set[docs]
    grad_set = grad_set[docs]
    att_grad_set = att_grad_set[docs]
    att_x_grad_set = att_x_grad_set[docs]
    actual_set = actual_set[docs]


    for _k_ in range(0,maximum):
        
        actual = actual_set.flatten().cpu().data.numpy()
     
        rand_pred = classification_report(actual, 
                                          rand_set[:,_k_].cpu().data.numpy(),
                                          output_dict = True)["macro avg"]["f1-score"]
        
        
        att_pred = classification_report(actual, 
                                          att_set[:,_k_].cpu().data.numpy(),
                                          output_dict = True)["macro avg"]["f1-score"]
        
        grad_pred = classification_report(actual, 
                                          grad_set[:,_k_].cpu().data.numpy(),
                                          output_dict = True)["macro avg"]["f1-score"]
        
        
        att_grad_pred = classification_report(actual, 
                                          att_grad_set[:,_k_].cpu().data.numpy(),
                                          output_dict = True)["macro avg"]["f1-score"]
        
        att_x_grad_pred = classification_report(actual, 
                                          att_x_grad_set[:,_k_].cpu().data.numpy(),
                                          output_dict = True)["macro avg"]["f1-score"]

        results["random"].append(rand_pred)
        results["attention"].append(att_pred)
        results["gradient"].append(grad_pred)
        results["grad_attention"].append(att_grad_pred)
        results["grad*attention"].append(att_x_grad_pred)
        
    results = pd.DataFrame.from_dict(results)
    
    results.plot(kind = "line", figsize = (18,10))
    
    ordering = "ascending"
    
    if largest:
        
        ordering = "descending"
    
    plt.savefig(save_path + "_correct_classified_" + ordering + ".png")
    
    results.to_csv(save_path + "_correct_classified_" + ordering + ".csv")
 