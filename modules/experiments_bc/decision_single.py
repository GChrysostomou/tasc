import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pandas as pd
from modules.experiments_bc.eval_utils import * 
from tqdm import trange

import json

with open('modules/config.txt', 'r') as f:
    args = json.load(f)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

"""MODEL OUT DISTRIBUTIONS"""

def effect_on_output(data, model, save_path):
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(100)
    np.random.seed(100)
    
    pbar = trange(len(data) * data.batch_size, desc='Most info. token', leave=True)
    
    results_flip = {}
    results_flip["max_source"] = []
    results_flip["grad"] = []
    results_flip["att_grad"] = []
    results_flip["att*grad"] = []
    results_flip["random_source"] = []
    results_flip["lengths"] = []

    if args["encoder"] == "bert":
        model.encoder.bert.embeddings.word_embeddings.weight.requires_grad_(True)
    else:
        model.encoder.embedding.weight.requires_grad_(True)

    
 
    for sentences, lengths, labels in data:
        torch.cuda.empty_cache()
        model.zero_grad()
        model.eval()
         
        if args["encoder"] == "bert":
            
            sentences, lengths, labels = torch.stack(sentences).transpose(0,1).to(device), lengths.to(device), labels.to(device)
                
        else:
            
            sentences, lengths, labels = Variable(sentences).to(device),Variable(lengths).to(device), Variable(labels).to(device)
        
        model.zero_grad()
 
        # original trained model    
        
        rows = torch.arange(sentences.size(0))
        
        model.train()
        
        yhat, weights = model(sentences, lengths, retain_gradient = True)
        
        yhat.max(-1)[0].sum().backward(retain_graph = True)
    
        g = model.encoder.embed.grad
        
        model.zero_grad()
        weights_def_grad = model.weights.grad
       
        with torch.no_grad():
        
            em = model.encoder.embed
    
            g1 = (g* em).sum(-1)
                               
            model.eval()
            
            """GRADIENT"""
            
            g_soft = g1.clone()
            
            g_soft.masked_fill_(model.masks.bool(),float("-inf"))
            
            sentence_grad = sentences.clone()
          
            sentence_grad[rows, g_soft.max(-1)[1]] = 0
          
            """RANDOM AT SOURCE"""
            
            ind_rand = torch.LongTensor(sentences.size(0)).random_(0, sentences.size(1))
            sentence_rand = sentences.clone()
           
            sentence_rand[rows, ind_rand] = 0
            
            
            """ MAX ATTENTION AT SOURCE """
            
            sentence_att = sentences.clone()
            
            sentence_att[:,:weights.size(1)][rows, weights.max(-1)[1]] = 0
    
            """ MAX ATTENTION GRAD AT SOURCE """
            
            weight_mul_grad = weights * weights_def_grad
            
            weights_def_grad.masked_fill_(model.masks[:,:max(lengths)].bool(),float("-inf"))
            
            sentence_att_grad = sentences.clone()[:,:max(lengths)]
            sentence_att_grad[rows, weights_def_grad.max(-1)[1]] = 0
            
            """ MAX ATTENTION * GRAD AT SOURCE """
       
            weight_mul_grad_soft = weight_mul_grad.clone()
            
            weight_mul_grad_soft.masked_fill_(model.masks[:,:max(lengths)].bool(),float("-inf"))
          
            sentence_att_mul_grad = sentences.clone()[:,:max(lengths)]
            sentence_att_mul_grad[rows, weight_mul_grad_soft.max(-1)[1]] = 0
       
            yhat_grad,_ = model(sentence_grad, lengths)
                        
            yhat_max_source, _ = model(sentence_att, lengths)
            
            yhat_rand_source, _ = model(sentence_rand, lengths)
            
            yhat_grad_att_source, _ = model(sentence_att_grad, lengths)
            
            yhat_grad_mul_att_source, _ = model(sentence_att_mul_grad, lengths)
            
  
            flip_rand_source = (yhat.max(-1)[1] != yhat_rand_source.max(-1)[1]).cpu().data.numpy()
            flip_grad = (yhat.max(-1)[1] != yhat_grad.max(-1)[1]).cpu().data.numpy()
            flip_max_source = (yhat.max(-1)[1] != yhat_max_source.max(-1)[1]).cpu().data.numpy()
            flip_grad_att = (yhat.max(-1)[1] != yhat_grad_att_source.max(-1)[1]).cpu().data.numpy()
            flip_grad_mul_att = (yhat.max(-1)[1] != yhat_grad_mul_att_source.max(-1)[1]).cpu().data.numpy()
    
            if len(yhat.shape) == 1:
                
                pass
                
            else:
                
           
                results_flip["random_source"].extend(flip_rand_source)
                results_flip["max_source"].extend(flip_max_source)
                results_flip["grad"].extend(flip_grad)
                results_flip["att_grad"].extend(flip_grad_att)
                results_flip["att*grad"].extend(flip_grad_mul_att)
                
                results_flip["lengths"].extend(lengths.cpu().data.numpy())
                
        pbar.update(data.batch_size)
        pbar.refresh()
           

    """Saving decision flip"""
    
    df = pd.DataFrame.from_dict(results_flip)

    df.to_csv(save_path + "_decision-flip-single.csv")
    
    summary = {}

    df = df.drop(columns = "lengths")
    
    for column in df.columns:
        
        sumation = 0
    
        summary[column] = {}
     
       
        for val, cnt in df[column].value_counts().iteritems():
     
            sumation += cnt
            
            summary[column][val] = cnt * 100 / df[column].value_counts().sum()
    

    summary = pd.DataFrame.from_dict(summary)
    
    summary = summary.T

    summary.to_csv(save_path + "_decision-flip-single-summary.csv")

