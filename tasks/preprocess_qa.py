import pickle
import glob
from torch.utils.data import DataLoader
import torch
import numpy as np
from torchtext.vocab import pretrained_aliases
import numpy as np

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

class pretrained_embeds():
    
    def __init__(self, model, ix_to_word):
        
        super(pretrained_embeds, self).__init__()
        
        self.vectors = pretrained_aliases[model](cache='../.vector_cache')

        self.ix_to_word = ix_to_word

        self.length = self.vectors.dim
                
    def processed(self):
        
        pretrained = np.zeros([len(self.ix_to_word), self.length])

        found = 0
        
        for i in range(pretrained.shape[0]):
            
            word = self.ix_to_word[i]
            
            if word in self.vectors.stoi.keys():
        
                pretrained[i,:] = self.vectors[word] 

                found += 1

            elif (word == "<PAD>") or (word == "<SOS>") or (word == "<EOS>"):
        
                pretrained[i,:] = np.zeros(self.length)
                
                found += 1
            
            else:
                
                pretrained[i,:] = np.random.randn(self.length)

        print("Found ", found, " words out of ", len(pretrained)) 
                
        return pretrained
    
    def save_processed(self, data, path):
        
        pickle.dump(data, open(path, "wb"))
        
    def load_processed(self, path):
        
        return pickle.load(open(path, "rb"))
    
def text_to_seq(doc, word_to_ix, answers = False):
    
    if answers:
        
        return word_to_ix[doc]
        
    else:
    
        return [word_to_ix[word] if word in word_to_ix else word_to_ix["<UNKN>"] for word in doc.split()]


def save_data(data, path):

    pickle.dump(data,  open( path, "wb" ))

def load_data(path):

    return pickle.load( open(path, "rb" ) )

import spacy, re
nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])

def cleaner(text, spacy=True) :
    text = re.sub(r'\s+', ' ', text.strip())
    if spacy :
        text = [t.text.lower() for t in nlp(text)]
    else :
        text = [t.lower() for t in text.split()]
    text = ['qqq' if any(char.isdigit() for char in word) else word for word in text]
    
    return text


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def extract_vocabulary_(min_df, dataframe):
        
    cvec = CountVectorizer(tokenizer=lambda text: text.split(" "), min_df=min_df, lowercase=False)
    to_transf = dataframe[dataframe.exp_split == "train"].paragraph.tolist() + dataframe[dataframe.exp_split == "train"].question.tolist()
    bow = cvec.fit_transform(to_transf)
    word_to_ix = cvec.vocabulary_
    
    for word in cvec.vocabulary_:
    
        word_to_ix[word] += 4

    word_to_ix["<PAD>"] = 0
    word_to_ix["<UNKN>"] = 1
    word_to_ix["<SOS>"] = 2
    word_to_ix["<EOS>"] = 3
    
    return word_to_ix
    
    

###########################################################################################
class DataHolder_QA():
    
    def __init__(self, train, dev, test, w2ix, w2ix_l, embeds = None):
        
        self.train = train
        self.dev = dev
        self.test = test
        self.w2ix = w2ix
        self.w2ix_l = w2ix_l
        
        if embeds is not None:
        
            self.pretrained_embeds = embeds
    

    
    
