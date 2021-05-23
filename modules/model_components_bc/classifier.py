import torch
import torch.nn as nn
import math 
import json 
from torch.autograd import Variable
from sklearn.metrics import *
from tqdm import trange

with open('modules/config.txt', 'r') as f:
    args = json.load(f)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):
    
    def __init__(self, encoder, attention,  mask_list,
                 hidden_dim, output_dim, tasc = None):
        super(Model, self).__init__()

        self.uniform = False
        self.output_dim = output_dim

        self.mask_list = mask_list

        self.encoder = encoder
        
        self.attention = attention
        
        self.tasc_mech = tasc

        self.output = nn.Linear(hidden_dim*2, output_dim)

        stdv = 1. / math.sqrt(hidden_dim*2)
        self.output.weight.data.uniform_(-stdv, stdv)
        self.output.bias.data.fill_(0)
        

    def forward(self, input, lengths, retain_gradient = False, ig = int(1)):
                
        self.hidden, last_hidden = self.encoder(input, lengths, ig = ig)

        if retain_gradient:

            self.encoder.embed.retain_grad()

        masks = 0
     
        for item in self.mask_list:
        
            masks += (input == item)
        
        self.masks = masks
        self.lengths = lengths
            
        self.weights = self.attention(self.hidden, masks[:,:max(lengths)])
                
        if retain_gradient:
            
            self.weights.retain_grad()
            
        if args["tasc"]:
            
            tasc_scores = self.tasc_mech(input, self.masks, lengths, self.encoder.embed)
            
            last_layer = (tasc_scores.unsqueeze(-1) * self.weights.unsqueeze(-1)*self.hidden).sum(1)
            
        else:
        
            last_layer = (self.weights.unsqueeze(-1)*self.hidden).sum(1)
 
        yhat = self.output(last_layer.squeeze(0))

        yhat = torch.softmax(yhat, dim = -1)
            
        
        return yhat.to(device), self.weights

    def get_omission_scores(self, input, lengths, predicted):

        input_pruned = input[:,:max(lengths)]

        omission_scores = []

        if len(predicted.shape) == 1:

            predicted = predicted.unsqueeze(0)

        predominant_class = predicted.max(-1)[1]
        self.eval()
        for _j in range(input_pruned.size(1)):
            torch.cuda.empty_cache()
            mask = torch.ones_like(input_pruned)
            mask[:,_j] = 0

            input_temp = input_pruned * mask

            ommited_pred = self.forward(input_temp, lengths)[0]

            if len(ommited_pred.shape) == 1:

                ommited_pred = ommited_pred.unsqueeze(0)

            ommited = ommited_pred[torch.arange(ommited_pred.size(0)), predominant_class]

            ommited[ommited != ommited] = 1

            scores = predicted.max(-1)[0] - ommited

            omission_scores.append(predicted.max(-1)[0] - ommited)

        omission_scores = torch.stack(omission_scores).transpose(0,1)

        return omission_scores

    def integrated_grads(self, original_input, original_grad, lengths, original_pred, steps = 20):

        grad_list = [original_grad]
        
        for x in torch.arange(start = 0.0, end = 1.0, step = (1.0-0.0)/steps):
            
            pred, _ = self.forward(original_input, lengths, retain_gradient = True, ig = x)
            
            if len(pred.shape) == 1:

                pred = pred.unsqueeze(0)

            rows = torch.arange(pred.size(0))

            if x == 0.0:

                baseline = pred[rows, original_pred[1]]

            pred[rows, original_pred[1]].sum().backward()

            g = self.encoder.embed.grad

            grad_list.append(g)

        attributions = torch.stack(grad_list).mean(0)

        em = self.encoder.embed

        ig = (attributions* em).sum(-1)[:,:max(lengths)]
        
        self.approximation_error = torch.abs((attributions.sum() - (original_pred[0] - baseline).sum()) / pred.size(0))

        return ig


def train(model, training, development, loss_function, optimiser, run,epochs = 10, cutoff = True, save_folder  = None, cutoff_len = 2):

    results = []
    
    results_for_run = ""
    
    cut_off_point = 0
    
    for epoch in trange(epochs, desc = "run {}:".format(run+1), maxinterval = 0.1):

        total_loss = 0
        
        for sentences, lengths, labels in training:
            
            model.zero_grad()
           
            if args["encoder"] == "bert":
            
                sentences, lengths, labels = torch.stack(sentences).transpose(0,1).to(device), lengths.to(device), labels.to(device)
                
            else:
                
                sentences, lengths, labels = Variable(sentences).to(device),Variable(lengths).to(device), Variable(labels).to(device)
                        
            yhat, weights =  model(sentences, lengths)
            
            if len(yhat.shape) == 1:
                
                yhat = yhat.unsqueeze(0)

            loss = loss_function(yhat, labels)
                              
            total_loss += loss.item()

            loss.backward()

            _, ind = torch.max(yhat, dim = 1)

            optimiser.step()

        dev_results, dev_loss = test(model, loss_function, development)    

        results.append([epoch, dev_results["macro avg"]["f1-score"], dev_loss, dev_results])
        
        
        results_for_run += "epoch - {} | train loss - {} | dev f1 - {} | dev loss - {} \n".format(epoch + 1,
                                    round(total_loss * training.batch_size / len(training),2),
                                    round(dev_results["macro avg"]["f1-score"], 3),
                                    round(dev_loss, 2))

        print(results_for_run)
        
        if save_folder is not None:
            
            if epoch == 0:
             
                torch.save(model.state_dict(), save_folder)
                
                saved_model_results  = dev_results
                saved_model_results["training_loss"] = total_loss * training.batch_size / len(training)
                saved_model_results["epoch"] = epoch+1
                saved_model_results["dev_loss"] = dev_loss
                
            else:
                
                if saved_model_results["dev_loss"] > dev_loss:
                  
                    torch.save(model.state_dict(), save_folder)
                
                    saved_model_results  = dev_results
                    saved_model_results["training_loss"] = total_loss * training.batch_size / len(training)
                    saved_model_results["epoch"] = epoch+1
                    saved_model_results["dev_loss"] = dev_loss

        ## cutoff
        if cutoff == True:
         
            if len(results) > cutoff_len:
         
                diff = results[-1][2] - results[-2][2]

                if diff > 0:
                    
                    cut_off_point += 1
                    
                else:
                    
                    cut_off_point = 0
                    
        if cut_off_point == cutoff_len:
            
            break
        
    return saved_model_results, results_for_run

def test(model, loss_function, data):
    
    predicted = [] 
    
    actual = []
    
    total_loss = 0
    
    model.eval()
    
    with torch.no_grad():

        for sentences, lengths, labels in data:
    
            if args["encoder"] == "bert":
            
                sentences, lengths, labels = torch.stack(sentences).transpose(0,1).to(device), lengths.to(device), labels.to(device)
                
            else:
                
                sentences, lengths, labels = Variable(sentences).to(device),Variable(lengths).to(device), Variable(labels).to(device)
    
            yhat, weights =  model(sentences, lengths)
            
            if len(yhat.shape) == 1:
    
                yhat = yhat.unsqueeze(0)
    
            loss = loss_function(yhat, labels)
        
            total_loss += loss.item()
            
            _, ind = torch.max(yhat, dim = 1)
    
            predicted.extend(ind.cpu().numpy())
    
            actual.extend(labels.cpu().numpy())
   
        results = classification_report(actual, predicted, output_dict = True)

    
    return results, (total_loss * data.batch_size / len(data)) 

