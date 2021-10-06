import torch
import matplotlib
matplotlib.use("Agg")
from torch.autograd import Variable
from modules.experiments_bc.eval_utils import * 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from tqdm import trange
import json
import os
import gc 

with open('modules/config.txt', 'r') as f:
    args = json.load(f)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def break_1_(data, model, save_path):

    pbar = trange(len(data) * data.batch_size, desc="extracting importance scores - 1", leave=True)

    fname = os.path.join(
        os.getcwd(), 
        "importance_scores", 
        save_path.split('/')[-2],
        ""
    )

    if os.path.exists(fname + f"{save_path.split('/')[-1]}-importance_scores.npy"):

        print(f"** Importance scores already extracted in -->> {fname}")
        return

    importance_scores = {}

    if args["encoder"] == "bert":

        model.encoder.bert.embeddings.requires_grad_(True)

    else:

        model.encoder.embedding.weight.requires_grad_(True)

    for inst_idx, sentences, lengths, labels in data:
        
        model.train() ## for gradients
        model.zero_grad()

        if args["encoder"] == "bert":
            
            sentences, lengths, labels = torch.stack(sentences).transpose(0,1).to(device), lengths.to(device), labels.to(device)
                
        else:
            
            sentences, lengths, labels = Variable(sentences).to(device),Variable(lengths).to(device), Variable(labels).to(device)

        for idx in inst_idx:
            
            importance_scores.update({idx:{}})

        original_prediction, weights_or = model(
            sentences, 
            lengths, 
            retain_gradient = True
        )

        original_prediction.max(-1)[0].sum().backward(retain_graph = True)

        g = model.encoder.embed.grad

        em = model.encoder.embed

        g1 = (g* em).sum(-1)[:,:max(lengths)].detach().cpu().numpy()
         
        weights_def_grad = model.weights.grad.detach().cpu().numpy()

        random = torch.randn(weights_or.shape).cpu().numpy()

        mask = model.masks[:,:max(lengths)].bool().cpu().detach().numpy()

        g1 = np.ma.filled(g1, mask)
        g1[g1 == 0] == float("-inf")

        weight_mul_grad = weights_or.cpu().detach().numpy() * weights_def_grad
        
        weights_def_grad = np.ma.filled(weights_def_grad, mask)
        weights_def_grad[weights_def_grad == 0] == float("-inf")

        weight_mul_grad = np.ma.filled(weight_mul_grad, mask)
        weight_mul_grad[weight_mul_grad == 0] == float("-inf")

        random = np.ma.filled(random, mask)
        random[random == 0] == float("-inf")


        for _i_, indx in enumerate(inst_idx):

            importance_scores[indx] = {
                "attention" : weights_or[_i_].detach().cpu().numpy(),
                "random" : random[_i_],
                "attention gradients" : weights_def_grad[_i_],
                "scaled attention" : weight_mul_grad[_i_],
                "gradients" : g1[_i_]

            }

        del weight_mul_grad
        del weights_def_grad
        del random
        del sentences
        del g1 
        del em
        del g
        del weights_or
        del original_prediction
        del lengths
        del labels
        del model.masks

        gc.collect()
        torch.cuda.empty_cache()


        pbar.update(data.batch_size)


    os.makedirs(fname, exist_ok = True)

    print(f"**** importance scores saved in ->> {fname}")
       
    np.save(fname + f"{save_path.split('/')[-1]}-importance_scores.npy", importance_scores)
    torch.cuda.empty_cache()
    return

def break_2_(data, model, save_path):

    pbar = trange(len(data) * data.batch_size, desc="extracting importance scores - IG", leave=True)

    fname = os.path.join(
        os.getcwd(), 
        "importance_scores", 
        save_path.split('/')[-2],
        ""
    )

    if os.path.exists(fname + f"{save_path.split('/')[-1]}-importance_scores.npy"):

        print(f"** Importance scores loaded from -->> {fname}")

        importance_scores = np.load(fname + f"{save_path.split('/')[-1]}-importance_scores.npy", allow_pickle = True).item()

    else:

        raise FileNotFoundError(
            """
            File does not exist in the path specified. Run def break_1_ first
            """
        )

    if args["encoder"] == "bert":

        model.encoder.bert.embeddings.requires_grad_(True)

    else:

        model.encoder.embedding.weight.requires_grad_(True)

    for inst_idx, sentences, lengths, labels in data:
        
        if "ig" in importance_scores[inst_idx[0]]:
            
            print("**** Already computed IG")
            return

        model.zero_grad()
        model.train() ## for gradients
        if args["encoder"] == "bert":
            
            sentences, lengths, labels = torch.stack(sentences).transpose(0,1).to(device), lengths.to(device), labels.to(device)
                
        else:
            
            sentences, lengths, labels = Variable(sentences).to(device),Variable(lengths).to(device), Variable(labels).to(device)

        original_prediction, weights_or = model(
            sentences, 
            lengths, 
            retain_gradient = True
        )

        original_prediction.max(-1)[0].sum().backward(retain_graph = True)

        g = model.encoder.embed.grad

        integrated_grads = model.integrated_grads(
            sentences, 
            g, 
            lengths, 
            original_pred = original_prediction.max(-1)
        )

        mask = model.masks[:,:max(lengths)].bool().cpu().detach().numpy()

        integrated_grads = np.ma.filled(integrated_grads, mask)
        integrated_grads[integrated_grads == 0] == float("-inf")
        
       
        for _i_, indx in enumerate(inst_idx):

            importance_scores[indx]["ig"] =  integrated_grads[_i_]

        del sentences
        del g
        del weights_or
        del original_prediction
        del lengths
        del labels
        del model.masks

        gc.collect()
        torch.cuda.empty_cache()


        pbar.update(data.batch_size)


    os.makedirs(fname, exist_ok = True)

    print(f"**** importance scores saved in ->> {fname}")
       
    np.save(fname + f"{save_path.split('/')[-1]}-importance_scores.npy", importance_scores)
    torch.cuda.empty_cache()
    return


def break_3_(data, model, save_path):

    pbar = trange(len(data) * data.batch_size, desc="extracting importance scores - Omission", leave=True)

    fname = os.path.join(
        os.getcwd(), 
        "importance_scores", 
        save_path.split('/')[-2],
        ""
    )

    if os.path.exists(fname + f"{save_path.split('/')[-1]}-importance_scores.npy"):

        print(f"** Importance scores loaded from -->> {fname}")

        importance_scores = np.load(fname + f"{save_path.split('/')[-1]}-importance_scores.npy", allow_pickle = True).item()

    else:

        raise FileNotFoundError(
            """
            File does not exist in the path specified. Run def break_1_ first
            """
        )

    if args["encoder"] == "bert":

        model.encoder.bert.embeddings.requires_grad_(True)

    else:

        model.encoder.embedding.weight.requires_grad_(True)

    for inst_idx, sentences, lengths, labels in data:
        
        model.zero_grad()
        model.eval()

        if args["encoder"] == "bert":
            
            sentences, lengths, labels = torch.stack(sentences).transpose(0,1).to(device), lengths.to(device), labels.to(device)
                
        else:
            
            sentences, lengths, labels = Variable(sentences).to(device),Variable(lengths).to(device), Variable(labels).to(device)

        original_prediction, weights_or = model(
            sentences, 
            lengths, 
            retain_gradient = False
        )

        mask = model.masks[:,:max(lengths)].bool().cpu().detach().numpy()

        omission_scores = model.get_omission_scores(
            sentences, 
            lengths, 
            original_prediction
        ).cpu().detach().numpy()

        mask = model.masks[:,:max(lengths)].bool().cpu().detach().numpy()
        omission_scores = np.ma.filled(omission_scores, mask)
        omission_scores[random == 0] == float("-inf")
       
        for _i_, indx in enumerate(inst_idx):

            importance_scores[indx]["omission"] =  omission_scores[_i_]

    
        del sentences
        del weights_or
        del original_prediction
        del lengths
        del labels
        del model.masks
        del omission_scores

        gc.collect()
        torch.cuda.empty_cache()

        pbar.update(data.batch_size)


    os.makedirs(fname, exist_ok = True)

    print(f"**** importance scores saved in ->> {fname}")
       
    np.save(fname + f"{save_path.split('/')[-1]}-importance_scores.npy", importance_scores)

    torch.cuda.empty_cache()
    return


def get_(data, model, save_path):

    pbar = trange(len(data) * data.batch_size, desc="extracting importance scores", leave=True)

    fname = os.path.join(
        os.getcwd(), 
        "importance_scores", 
        save_path.split('/')[-2],
        ""
    )

    if os.path.exists(fname + f"{save_path.split('/')[-1]}-importance_scores.npy"):

        print(f"** Importance scores already extracted in -->> {fname}")
        return

    importance_scores = {}

    if args["encoder"] == "bert":

        model.encoder.bert.embeddings.requires_grad_(True)

    else:

        model.encoder.embedding.weight.requires_grad_(True)

    for inst_idx, sentences, lengths, labels in data:
        
        model.zero_grad()

        if args["encoder"] == "bert":
            
            sentences, lengths, labels = torch.stack(sentences).transpose(0,1).to(device), lengths.to(device), labels.to(device)
                
        else:
            
            sentences, lengths, labels = Variable(sentences).to(device),Variable(lengths).to(device), Variable(labels).to(device)

        for idx in inst_idx:
            
            importance_scores.update({idx:{}})

        original_prediction, weights_or = model(
            sentences, 
            lengths, 
            retain_gradient = True
        )

        original_prediction.max(-1)[0].sum().backward(retain_graph = True)

        g = model.encoder.embed.grad

        em = model.encoder.embed

        g1 = (g* em).sum(-1)[:,:max(lengths)].detach().cpu().numpy()
         
        weights_def_grad = model.weights.grad.detach().cpu().numpy()

        random = torch.randn(weights_or.shape).cpu().numpy()

        integrated_grads = model.integrated_grads(
            sentences, 
            g, 
            lengths, 
            original_pred = original_prediction.max(-1)
        )
        
        omission_scores = model.get_omission_scores(
            sentences, 
            lengths, 
            original_prediction
        ).cpu().detach().numpy()

        mask = model.masks[:,:max(lengths)].bool().cpu().detach().numpy()

        g1 = np.ma.filled(g1, mask)
        g1[g1 == 0] == float("-inf")

        integrated_grads = np.ma.filled(integrated_grads, mask)
        integrated_grads[integrated_grads == 0] == float("-inf")

        weight_mul_grad = weights_or.cpu().detach().numpy() * weights_def_grad
        
        weights_def_grad = np.ma.filled(weights_def_grad, mask)
        weights_def_grad[weights_def_grad == 0] == float("-inf")

        weight_mul_grad = np.ma.filled(weight_mul_grad, mask)
        weight_mul_grad[weight_mul_grad == 0] == float("-inf")

        random = np.ma.filled(random, mask)
        random[random == 0] == float("-inf")

        omission_scores = np.ma.filled(omission_scores, mask)
        omission_scores[random == 0] == float("-inf")

        for _i_, indx in enumerate(inst_idx):

            importance_scores[indx] = {
                "attention" : weights_or[_i_].detach().cpu().numpy(),
                "random" : random[_i_],
                "omission" : omission_scores[_i_],
                "attention gradients" : weights_def_grad[_i_],
                "scaled attention" : weight_mul_grad[_i_],
                "ig" : integrated_grads[_i_],
                "gradients" : g1[_i_]

            }

    
        del integrated_grads
        del weight_mul_grad
        del weights_def_grad
        del random
        del omission_scores
        del sentences
        del g1 
        del em
        del g
        del weights_or
        del original_prediction
        del lengths
        del labels
        del model.masks

        gc.collect()
        torch.cuda.empty_cache()

        pbar.update(data.batch_size)


    os.makedirs(fname, exist_ok = True)

    print(f"**** importance scores saved in ->> {fname}")
       
    np.save(fname + f"{save_path.split('/')[-1]}-importance_scores.npy", importance_scores)

    return