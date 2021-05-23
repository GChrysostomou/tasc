#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch import optim
import json 
import numpy as np
import pandas as pd
from tqdm import tqdm


with open('modules/config.txt', 'r') as f:
    args = json.load(f)

if args["mechanism"] == "dot":
    
    from modules.model_components_bc.attentions import DotAttention as attention_mech
    
else:
    
    from modules.model_components_bc.attentions import TanhAttention as attention_mech

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from modules.model_components_bc.encoder import Encoder
from modules.model_components_bc.classifier import train, test
from modules.model_components_bc import tasc

from modules.run_experiments import *
from modules.model_components_bc.classifier import Model


def optimiser_fun(model, encoder):
    
    optimiser = getattr(torch.optim, args["optimiser"]) 

    if args["encoder"] == "bert":
        
        optimiser = optim.Adam([
                    {'params': model.encoder.parameters(), 'lr': 1e-5},
                    {'params': model.output.parameters(), 'lr': 1e-4},
                    {'params': model.attention.parameters(), 'lr': 1e-4}
                ], amsgrad = True, weight_decay = 10e-5)

        if args["tasc"]:
            
            optimiser = optim.Adam([
                {'params': model.encoder.parameters(), 'lr': 1e-5},
                {'params': model.output.parameters(), 'lr': 1e-4},
                {'params': model.attention.parameters(), 'lr':1e-4},
                {'params':model.tasc_mech.parameters(), 'lr':1e-4}
            ], amsgrad = True, weight_decay = 10e-5)

            if args["dataset"] == "mimicanemia":
                
                optimiser = optim.Adam([
                        {'params': model.encoder.parameters(), 'lr': 1e-5},
                        {'params': model.output.parameters(), 'lr': 1e-4},
                        {'params': model.attention.parameters(), 'lr':1e-4},
                        {'params':model.tasc_mech.parameters(), 'lr':1e-4}
                     ], amsgrad = True, weight_decay = 10e-5)

    
    else:
        
        optimiser = optim.Adam([ param for param in model.parameters() if param.requires_grad == True],
                                   amsgrad = True, 
                                   weight_decay = 10e-5)
        
    return optimiser
        
        

def train_binary_classification_model(data):
    
    
    """
    Trains models and monitors on dev set
    Also produces statistics for each run (seed)    
    """
    run_train = 0
    for number in range(len(args["seeds"])):
        
        torch.manual_seed(args["seeds"][number])
        np.random.seed(args["seeds"][number])
        
  
          
        attention = attention_mech(args["hidden_dim"])
                  
        encoder = Encoder(embedding_dim=args["embedding_dim"],
                        vocab_size=data.vocab_size,
                         hidden_dim=args["hidden_dim"], 
                         encode_sel = args["encoder"],
                     embedding = data.pretrained_embeds)
        
        if args["tasc"]:
            
            tasc_variant = getattr(tasc, args["tasc"]) 
            
            tasc_mech = tasc_variant(args["vocab_size"], args["seeds"][number])
            
        else:
            
            tasc_mech = None
            
    
        model = Model(encoder = encoder, 
                      attention = attention, 
                        mask_list=data.mask_list,
                     hidden_dim=args["hidden_dim"],
                     output_dim=data.output_size, tasc = tasc_mech)
        
        model.to(device)
        
        loss_function = nn.CrossEntropyLoss()
        
        if args["encoder"] == "bert":
        
            model.encoder.bert.embeddings.word_embeddings.weight.requires_grad = False
    
            optimiser = optimiser_fun(model, args["encoder"])
            
            total_params = sum(p.numel() for p in model.parameters())
            total_trainable_params = sum(p.numel() for p in model.parameters()
                                         if p.requires_grad)
    
            print("Total Params:", total_params)
            print("Total Trainable Params:", total_trainable_params)
            
            assert (total_params - total_trainable_params) == model.encoder.bert.embeddings.word_embeddings.weight.numel()
        else:
            
            model.encoder.embedding.weight.requires_grad = False
            
            optimiser = optimiser_fun(model, args["encoder"])
           
            total_params = sum(p.numel() for p in model.parameters())
            total_trainable_params = sum(p.numel() for p in model.parameters()
                                         if p.requires_grad)
        
            print("Total Params:", total_params)
            print("Total Trainable Params:", total_trainable_params)
            assert (total_params - total_trainable_params) == model.encoder.embedding.weight.numel()
    
        save_folder = args["save_path"] + args["encoder"] + "_" + args["mechanism"] + str(number) + ".model"
  
        dev_results, results_to_save = train(model,  
              data.training, 
              data.development, 
              loss_function,
            optimiser,
            epochs = args["epochs"],
              cutoff = False, 
              save_folder = save_folder,
              run = run_train)
        
        text_file = open(args["save_path"]  +"model_run_stats/" + args["encoder"] + "_" + args["mechanism"] + "_run_" + str(run_train + 1) + ".txt", "w")
        text_file.write(results_to_save)
        text_file.close()
        
        
        run_train +=1
        
        
        
        df = pd.DataFrame(dev_results)
        df.to_csv(args["save_path"]  +"model_run_stats/" + args["encoder"] + "_" + args["mechanism"] + "_best_model_devrun:" + str(number) + ".csv")
        
import glob
import os 
def evaluate_trained_bc_model(data):    
    
    """
    Runs trained models on test set
    Also keeps the best model for experimentation
    and produces statistics    
    """
    
    saved_models = glob.glob(args["save_path"] + "*.model")
    
    stats_report = {}
 
    stats_report[args["mechanism"]] = {}
                

    for j in range(len(args["seeds"])):
        
        torch.manual_seed(args["seeds"][j])
        np.random.seed(args["seeds"][j])
  
          
        attention = attention_mech(args["hidden_dim"])
                  
        encoder = Encoder(embedding_dim=args["embedding_dim"],
                        vocab_size=data.vocab_size,
                         hidden_dim=args["hidden_dim"], 
                         encode_sel = args["encoder"],
                     embedding = data.pretrained_embeds)
        
        if args["tasc"]:
            
            tasc_variant = getattr(tasc, args["tasc"]) 
            
            tasc_mech = tasc_variant(args["vocab_size"], args["seeds"][j])
            
        else:
            
            tasc_mech = None
            
    
        model = Model(encoder = encoder, 
                      attention = attention, 
                        mask_list=data.mask_list,
                     hidden_dim=args["hidden_dim"],
                     output_dim=data.output_size, 
                     tasc = tasc_mech)
        
        model.to(device)
        
        
        current_model = args["save_path"] + args["encoder"] + "_" + args["mechanism"] + str(j) + ".model"
       
        index_model = saved_models.index(current_model)
        
        # loading the trained model
    
        model.load_state_dict(torch.load(saved_models[index_model], map_location=device))
        
        model.to(device)
        
        loss_function = nn.CrossEntropyLoss()

        test_results,test_loss = test(model, loss_function, data.testing)
        
        df = pd.DataFrame(test_results)
       
        df.to_csv(args["save_path"]  +"/model_run_stats/" + args["encoder"] + "_" + args["mechanism"] + "_best_model_testrun:" + str(j) + ".csv")
     
        stats_report[args["mechanism"]]["Macro F1 - avg:run:" +str(j)] = test_results["macro avg"]["f1-score"]
        
             
        print("Run: ", j, 
              " Test loss: ", round(test_loss), 
              " Test accuracy: ", round(test_results["macro avg"]["f1-score"], 3),
             )
    
    
    
    """
    now to keep only the best model
    
    """
    
    performance_list = tuple(stats_report[args["mechanism"]].items()) ## keeping the runs and acuracies
    
    performance_list = [(x.split(":")[-1], y) for (x,y) in performance_list]
    
    sorted_list = sorted(performance_list, key = lambda x: x[1])
    
    models_to_get_ridoff, _ = zip(*sorted_list[:len(args["seeds"]) - 1])
    
    for item in models_to_get_ridoff:
        
        os.remove(args["save_path"] + args["encoder"] + "_" + args["mechanism"]  + str(item) + ".model")
    
    """
    saving the stats
    """
    
    stats_report[args["mechanism"]]["mean"] = np.asarray(list(stats_report[args["mechanism"]].values())).mean()
    stats_report[args["mechanism"]]["std"] = np.asarray(list(stats_report[args["mechanism"]].values())).std()
    
    df = pd.DataFrame(stats_report)
    df.to_csv(args["save_path"] + args["encoder"] + "_" + args["mechanism"] + "_predictive_performances.csv")
    


def conduct_experiments(data):    
    
    """
    Runs trained models on test set
    Also keeps the best model for experimentation
    and produces statistics    
    """
    
    saved_models = glob.glob(args["save_path"] + "*.model")
    
    no_number_models = [x.lower() for x in saved_models]

    no_number_models = [(x.split(".model")[0][:-1] + ".model") for x in saved_models if ".model" in x]
    
    stats_report = {}
 
    stats_report[args["mechanism"]] = {}
                
    torch.manual_seed(24)
    np.random.seed(24)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
  
    attention = attention_mech(args["hidden_dim"])
              
    encoder = Encoder(embedding_dim=args["embedding_dim"],
                    vocab_size=data.vocab_size,
                     hidden_dim=args["hidden_dim"], 
                     encode_sel = args["encoder"],
                 embedding = data.pretrained_embeds)
    
    if args["tasc"]:
        
        tasc_variant = getattr(tasc, args["tasc"]) 
        
        tasc_mech = tasc_variant(args["vocab_size"], args["seeds"][0])
        
    else:
        
        tasc_mech = None
        

    model = Model(encoder = encoder, 
                  attention = attention, 
                    mask_list=data.mask_list,
                 hidden_dim=args["hidden_dim"],
                 output_dim=data.output_size, 
                 tasc = tasc_mech)
    
    model.to(device)
    
    
    current_model = args["save_path"] + args["encoder"] + "_" + args["mechanism"] + ".model"
   
    index_model = no_number_models.index(current_model)
    
    # loading the trained model

    model.load_state_dict(torch.load(saved_models[index_model], map_location=device))
    
    model.to(device)
    
    evaluation = evaluate(classifier = model, 
                      loss_function = nn.CrossEntropyLoss(),
                      data = [args["dataset"], data],
                      save_path = args["experiments_path"]
                      )
    
    evaluation.decision_flip_single()
        
    evaluation.decision_flip_set()
        
    torch.cuda.empty_cache()
