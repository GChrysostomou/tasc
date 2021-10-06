import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader
import pandas as pd
import json

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def padder(ix_document, pad_len):
    
    seq_lens = []
    
    padded_all = []
    
    for doc in ix_document:
        
        if len(doc) == 0 :
           
            pass
               
        if (len(doc) < pad_len) & (len(doc) > 0):

            length = len(doc)
            
            diff = pad_len - len(doc)

            add_pad = [0]*diff

            padded = doc + add_pad

        elif len(doc) == pad_len:

            padded = doc
            
            length = len(doc)

        elif len(doc) > pad_len:

            padded = doc[:pad_len]
            
            length = pad_len
            
        padded_all.append(padded)
        seq_lens.append(length)
        
    return padded_all, seq_lens


class dataholder():
    
    def __init__(self, directory, dataset, B_SIZE  = 32):
        
        """
        Dataholder class (for non-bert instances)
        Accepts as input the data directory : directory
        The dataset : dataset
        and batch size: B_SIZE
        """
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        
        self.directory = directory
        self.dataset = dataset
        self.batch_size = B_SIZE
        self.hidden_dim = 64
        self.embedding_dim = 300
    
        all_data = pickle.load(open(directory + dataset + "/data.p", "rb"))
    
        self.w2ix = all_data.w2ix
        self.vocab_size = len(self.w2ix)    
        
        self.mask_list = []
        self.mask_tokens = ["<PAD>", "<SOS>", "<EOS>", "."]
        
        for item in self.mask_tokens:
            
            if item in self.w2ix:
                
                self.mask_list.append(self.w2ix[item])
        
        self.pretrained_embeds = all_data.pretrained_embeds
        
        
        # In[4]:
        
        
        tr_idx, x_train, y_train = zip(*all_data.train)
        dev_idx, x_dev, y_dev = zip(*all_data.dev)
        test_idx, x_test, y_test = zip(*all_data.test)
        
        print("\nVocab size:", len(self.w2ix),
                "\nTraining size:", len(y_train),
                "\nDev size:", len(y_dev),
                "\nTest size:", len(y_test))
        
        # In[5]:
        
        self.output_size= len(np.unique(y_train))
        
        print("\nOutput dimension: ", self.output_size, "\n")
        
        
        self.sequence_length = all_data.sequence_length()
        
        if dataset == "mimicanemia":
        
        	self.sequence_length = 2200
        
        print("--Sequence length :", self.sequence_length, "\n")
        
        from modules.utils import padder
        
        x_train_pad, train_lengths = padder(x_train, pad_len = self.sequence_length)
        x_dev_pad, dev_lengths = padder(x_dev, pad_len = self.sequence_length)
        x_test_pad, test_lengths = padder(x_test, pad_len = self.sequence_length)
        
        
        # In[11]:
        
        x_train_pad = torch.LongTensor(x_train_pad)#.to(device)
        x_dev_pad = torch.LongTensor(x_dev_pad)#.to(device)
        x_test_pad = torch.LongTensor(x_test_pad)#.to(device)
        train_lengths = torch.LongTensor(train_lengths)#.to(device)
        dev_lengths =  torch.LongTensor(dev_lengths)#.to(device)
        test_lengths = torch.LongTensor(test_lengths)#.to(device)
        y_train = torch.LongTensor(y_train)#.to(device)
        y_dev = torch.LongTensor(y_dev)#.to(device)
        y_test = torch.LongTensor(y_test)#.to(device)
        
        
        # In[12]:
        
        training_prebatch = list(zip(tr_idx, x_train_pad, train_lengths, y_train))
        dev_prebatch = list(zip(dev_idx, x_dev_pad, dev_lengths, y_dev))
        testing_prebatch = list(zip(test_idx, x_test_pad, test_lengths, y_test))
        
        
        training_prebatch = sorted(training_prebatch, key = lambda x : x[2], reverse = False)
        dev_prebatch = sorted(dev_prebatch, key = lambda x : x[2], reverse = False)
        testing_prebatch = sorted(testing_prebatch, key = lambda x : x[2], reverse = False)
        
        # In[13]:
        
        ### removing sos and eos only sentences
        
        train_prebatch = [x for x in training_prebatch if x[2] > 2]
        dev_prebatch = [x for x in dev_prebatch if x[2] > 2]
        test_prebatch = [x for x in testing_prebatch if x[2] > 2]
        
    
        self.training = DataLoader(train_prebatch, batch_size = self.batch_size, 
                                  shuffle = True, pin_memory = False)
            
        self.development = DataLoader(dev_prebatch, batch_size = self.batch_size, 
                               shuffle = False, pin_memory = False)
        
        
        self.testing = DataLoader(test_prebatch, batch_size = self.batch_size, 
                               shuffle = False, pin_memory = False)
        


def bertify(x, not_include = ["<SOS>", "<EOS>"]):
    
    bertification = []
    
    for word in x.split():
        
        if word == "<UNKN>":
            
            word = '[UNK]'
            
            bertification.append(word)
            
        elif word in not_include:
            
            pass
     
        else:
        
            bertification.append(word)
            
    return " ".join(bertification)
 
def bert_padder(x, max_len):
        
    if len(x) < max_len:
    
        x += [0]*(int(max_len) - len(x))
    
    elif len(x) > max_len:

        x = x[:max_len - 1]
        x += [102]

    return x

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def mimic_halfer(x):

    stopwordlist = stopwords.words("english") + ["qqq", "<DE>", ":", ")", "(", ".", "/"]

    cut_no_stop = [word for word in x.split() if not word in stopwordlist]
    
    revised = cut_no_stop[20:276] + cut_no_stop[-256:]

    return " ".join(revised).lower()
        
from tqdm.auto import tqdm
from transformers import AutoTokenizer
       
class dataholder_bert():
    
    def __init__(self, directory, dataset, B_SIZE  = 8, bert_model = "bert-base_uncased"):        
        
        self.directory = directory
        self.dataset = dataset
        self.batch_size = B_SIZE
        self.hidden_dim = 768 // 2
        self.embedding_dim = None
        self.pretrained_embeds = None
        self.mask_list = [101, 102, 0]
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        
        all_data = pd.read_csv(directory + dataset + "/" + dataset + "_dataset.csv")

        all_data["text"] = all_data["text"].astype(str)

        self.output_size = 2
        
        print("\nOutput dimension: ", self.output_size, "\n")
        
        pretrained_weights = bert_model
       
        if dataset == "mimicanemia": bert_max_length = 512
        else: bert_max_length = 512
        
        tokenizer = AutoTokenizer.from_pretrained(pretrained_weights, attention_window = bert_max_length)
        
        self.vocab_size = tokenizer.vocab_size

        # treat mimic for long text
        if dataset == "mimicanemia":
            # mimic requires reshuffling
            all_data = all_data.sample(frac = 1.0, random_state = 100)
            # borrowed idea from Fine-tune BERT for Text Classification 
            all_data["text"] = all_data["text"].apply(lambda x: mimic_halfer(x))
        
        #remove sos and eos and replace unkn with bert symbols
        all_data["text"] = all_data["text"].apply(lambda x: bertify(x))
        # tokenize to maximum length the sequences and add the CLS token and ?SEP? at the enD??
        
        tqdm.pandas(desc="tokenizing")
        all_data["text"] = all_data["text"].progress_apply(lambda x: tokenizer.encode_plus(x, max_length = bert_max_length,truncation = True)["input_ids"])
        all_data["lengths"] = all_data["text"].apply(lambda x: len(x)) 
        
        self.sequence_length = int(all_data.lengths.quantile(q = 0.95))
            
        if self.sequence_length  < 50:
            
            self.sequence_length  = 50
        
        if dataset == "mimicanemia":
        
        	self.sequence_length  = 512
        
        
        print("--Sequence length :", self.sequence_length , "\n")
        
        if self.sequence_length  < 512:
            
            bert_max_length = self.sequence_length             
        
        all_data["text_encoded"] = all_data["text"].apply(lambda x: bert_padder(x, bert_max_length))

        train_prebatch = all_data[all_data.exp_split == "train"][["instance_idx", "text_encoded", "lengths", "label"]].values.tolist()
        dev_prebatch = all_data[all_data.exp_split == "dev"][['instance_idx', "text_encoded", "lengths", "label"]].values.tolist()
        test_prebatch = all_data[all_data.exp_split == "test"][['instance_idx', "text_encoded", "lengths", "label"]].values.tolist()
        
        # ### keep non zero sequences
        train_prebatch = sorted(train_prebatch, key = lambda x : x[2], reverse = False)
        dev_prebatch = sorted(dev_prebatch, key = lambda x: x[2], reverse = False)
        test_prebatch = sorted(test_prebatch, key = lambda x: x[2], reverse = False)
        
        ### removing sos and eos only sentences
        train_prebatch = [x for x in train_prebatch if x[2] > 2]
        dev_prebatch = [x for x in dev_prebatch if x[2] > 2]
        test_prebatch = [x for x in test_prebatch if x[2] > 2]
        


        self.training = DataLoader(train_prebatch, batch_size = self.batch_size, 
                      shuffle = False, pin_memory = False)
        
        self.development = DataLoader(dev_prebatch, batch_size = self.batch_size, 
                               shuffle = False, pin_memory = False)
        
        
        self.testing = DataLoader(test_prebatch, batch_size = self.batch_size, 
                               shuffle = False, pin_memory = False)


"""
Faithfulness metrics
"""

def sufficiency_(full_text_probs : np.array, reduced_probs : np.array) -> np.array:

    sufficiency = 1 - np.maximum(0, full_text_probs - reduced_probs)

    return sufficiency

def normalized_sufficiency_(model, original_sentences : torch.tensor, rationale_mask : torch.tensor, 
                            inputs : dict, full_text_probs : np.array, full_text_class : np.array, rows : np.array, 
                            suff_y_zero : np.array) -> np.array:

    ## for sufficiency we always keep the rationale
    ## since ones represent rationale tokens
    ## preserve cls
    rationale_mask[:,0] = 1
    ## preserve sep
    rationale_mask[torch.arange(rationale_mask.size(0)).to(device), inputs["lengths"]-1] = 1

    inputs["input"]  =  rationale_mask[:,:max(inputs["lengths"])].long() * original_sentences

    yhat, _  = model(**inputs)

    yhat = torch.softmax(yhat.detach().cpu(), dim = -1).numpy()

    if type(rows) != np.ndarray:

        reduced_probs = yhat[full_text_class]

    else:

        reduced_probs = yhat[rows, full_text_class]

    ## reduced input sufficiency
    suff_y_a = sufficiency_(full_text_probs, reduced_probs)
    suff_y_a = np.nan_to_num(suff_y_a, nan=1.)
    # return suff_y_a
    suff_y_zero -= 1e-4 ## to avoid nan

    norm_suff = np.maximum(0, (suff_y_a - suff_y_zero) / (1 - suff_y_zero))

    norm_suff = np.clip( norm_suff, a_min = 0, a_max = 1)

    return norm_suff, reduced_probs

def comprehensiveness_(full_text_probs : np.array, reduced_probs : np.array) -> np.array:

    comprehensiveness = np.maximum(0, full_text_probs - reduced_probs)

    return comprehensiveness

def normalized_comprehensiveness_(model, original_sentences : torch.tensor, rationale_mask : torch.tensor, 
                                  inputs : dict, full_text_probs : np.array, full_text_class : np.array, rows : np.array, 
                                  suff_y_zero : np.array) -> np.array:
    
    ## for comprehensivness we always remove the rationale and keep the rest of the input
    ## since ones represent rationale tokens, invert them and multiply the original input
    rationale_mask = (rationale_mask == 0).int()

    inputs["input"] =  original_sentences * rationale_mask[:,:max(inputs["lengths"])].long()
    
    yhat, _  = model(**inputs)

    yhat = torch.softmax(yhat, dim = -1).detach().cpu().numpy()

    if type(rows) != np.ndarray:
        
        reduced_probs = yhat[full_text_class]

    else:

        reduced_probs = yhat[rows, full_text_class]

     ## reduced input sufficiency
    comp_y_a = comprehensiveness_(full_text_probs, reduced_probs)
    comp_y_a = np.nan_to_num(comp_y_a, nan=1.)
    # return comp_y_a
    suff_y_zero -= 1e-4 # to avoid nan

    ## 1 - suff_y_0 == comp_y_1
    norm_comp = np.maximum(0, comp_y_a / (1 - suff_y_zero))

    norm_comp = np.clip(norm_comp, a_min = 0, a_max = 1)

    return norm_comp, yhat


"""MASK FUNCTIONS"""


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def create_rationale_mask_(
        importance_scores = torch.tensor, 
        no_of_masked_tokens = np.ndarray,
        method = "topk", batch_input_ids = None
    ):

    rationale_mask = []

    for _i_ in range(importance_scores.size(0)):
        
        score = importance_scores[_i_]
        tokens_to_mask = int(no_of_masked_tokens[_i_])
        
        ## if contigious or not a unigram (unigram == topk of 1)
        if method == "contigious" and tokens_to_mask > 1:

            top_k = contigious_(
                importance_scores = score,
                tokens_to_mask = tokens_to_mask
            )
        
        else:

            top_k = topk_(
                importance_scores = score,
                tokens_to_mask = tokens_to_mask
            )

        ## create the instance specific mask
        ## 1 represents the rationale :)
        ## 0 represents tokens that we dont care about :'(
        mask = torch.zeros(score.shape).to(device)
        mask = mask.scatter_(-1,  top_k.to(device), 1).long()

        rationale_mask.append(mask)

    rationale_mask = torch.stack(rationale_mask).to(device)

    return rationale_mask

def contigious_(importance_scores, tokens_to_mask):

    ngram = torch.stack([importance_scores[i:i + tokens_to_mask] for i in range(len(importance_scores) - tokens_to_mask + 1)])
    indxs = [torch.arange(i, i+tokens_to_mask) for i in range(len(importance_scores) - tokens_to_mask + 1)]
    top_k = indxs[ngram.sum(-1).argmax()]

    return top_k

def topk_(importance_scores, tokens_to_mask):

    top_k = torch.topk(importance_scores, tokens_to_mask).indices

    return top_k

def batch_from_dict_(inst_indx, metadata, target_key = "attention"):

    new_tensor = []

    for _id_ in inst_indx:

        new_tensor.append(
            metadata[_id_][target_key]
        )

    return torch.tensor(new_tensor).to(device)