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

def ixed_text(data, word_to_ix):
                                                                                            
    ixed_all = []

    for doc in data:

        text, lab = doc[0], doc[1]

        ixed = [word_to_ix[word] if word in word_to_ix else word_to_ix["<UNKN>"] for word in text]

        ixed = [ixed, lab]

        ixed_all.append(ixed)

    return ixed_all

def ixed_text_mt(data, word_to_ix_s, word_to_ix_t):

    ixed_all = []

    for doc in data:

        ixed = [([word_to_ix_s[word] for word in en.split()]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                , [word_to_ix_t[words] for words in de.split()]) for (en, de) in data]

        ixed_all.append(ixed)

    return ixed_all

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
    
    return " ".join(text).split()


class DataHolder_BC():
    
    def __init__(self, train, dev, test, w2ix, embeds = None):
        
        self.train = train
        self.dev = dev
        self.test = test
        self.w2ix = w2ix
        
        if embeds is not None:
        
            self.pretrained_embeds = embeds
            
            
    def sequence_length(self):
        
        x_train, y_train = zip(*self.train)
        
        max_len = [len(x) for x in x_train]

        counts, bins = np.histogram(max_len, bins = 100)
        
        total = sum(counts)
        
        temp = 0
        
        for i, count in enumerate(counts):
            
            temp += count
            
            if temp >= total*0.95:
                
                self.sequence_length = int(bins[i])
                
                break
        
        if self.sequence_length < 50:
            
            self.sequence_length = max([len(x) for x in x_train])
    
        
        return self.sequence_length
    
    
class DataHolder_QA():
    
    def __init__(self, train, dev, test, w2ix, w2ix_l, embeds = None):
        
        self.train = train
        self.dev = dev
        self.test = test
        self.w2ix = w2ix
        self.w2ix_l = w2ix_l
        
        if embeds is not None:
        
            self.pretrained_embeds = embeds
    
class DataHolder_MT():
    
    def __init__(self, train, dev, test, w2ix_s, w2ix_t, embeds = None):
        
        self.train = train
        self.dev = dev
        self.test = test
        self.w2ix_s = w2ix_s
        self.w2ix_t = w2ix_t
        
        if embeds is not None:
        
            self.pretrained_embeds = embeds 
            
        else:
            
            self.pretrained_embeds = None
            
    def _sequence_length_(self):
        
        x_train, y_train = zip(*self.train)
        
        max_len = [len(x) for x in x_train]

        counts, bins = np.histogram(max_len, bins = 100)
        
        total = sum(counts)
        
        temp = 0
        
        for i, count in enumerate(counts):
            
            temp += count
            
            if temp >= total*1.0:
                
                self.sequence_length = int(bins[i])
                
                break
        
        return self.sequence_length
    
    @property
    def _stats_(self):
        
        stats = {}
        stats["train_size"] = len(self.train)
        stats["dev_size"] = len(self.dev)
        stats["test_size"] = len(self.test)
        stats["source_vocab_size"] = len(self.w2ix_s)
        stats["target_vocab_size"] = len(self.w2ix_t)
        stats["average_length_seq"] = sum([len(x) for x,y in self.train])/len(self.train)
        
        return stats
        
    def _pad_(self):
        
        self.x_train, self.y_train = zip(*self.train)
        self.x_dev, self.y_dev = zip(*self.dev)
        self.x_test, self.y_test = zip(*self.test)
        
        self.x_train_pad, self.train_lengths = padder(self.x_train, pad_len = self._sequence_length_())
        self.x_dev_pad, self.dev_lengths = padder(self.x_dev, pad_len = self._sequence_length_())
        self.x_test_pad, self.test_lengths = padder(self.x_test, pad_len = self._sequence_length_())

        self.y_train_pad, self.y_train_lengths = padder(self.y_train, pad_len = self._sequence_length_())
        self.y_dev_pad, self.y_dev_lengths = padder(self.y_dev, pad_len = self._sequence_length_())
        self.y_test_pad, self.y_test_lengths = padder(self.y_test, pad_len = self._sequence_length_())
    
    @property
    def _to_tensor_(self):
        
        self.x_train_pad = torch.LongTensor(self.x_train_pad)#.to(device)
        self.x_dev_pad = torch.LongTensor(self.x_dev_pad)#.to(device)
        self.x_test_pad = torch.LongTensor(self.x_test_pad)#.to(device)
        self.train_lengths = torch.LongTensor(self.train_lengths)#.to(device)
        self.dev_lengths =  torch.LongTensor(self.dev_lengths)#.to(device)
        self.test_lengths = torch.LongTensor(self.test_lengths)#.to(device)

        self.y_train_pad = torch.LongTensor(self.y_train_pad)#.to(device)
        self.y_dev_pad = torch.LongTensor(self.y_dev_pad)#.to(device)
        self.y_test_pad = torch.LongTensor(self.y_test_pad)#.to(device)
        self.y_train_lengths = torch.LongTensor(self.y_train_lengths)#.to(device)
        self.y_dev_lengths = torch.LongTensor(self.y_dev_lengths)#.to(device)
        self.y_test_lengths = torch.LongTensor(self.y_test_lengths)#.to(device)
        
    @property
    def _batch_size_(self, batch_size = 32):
        
        self.batch_size = batch_size
        
        return self.batch_size
    
                
    def _entity_mask_(self, x, mask_token_list = ["<PAD>"]):
    
        mask_token_list = ["<PAD>"]
        
        assert self.w2ix_s["<PAD>"] == self.w2ix_t["<PAD>"]
        
        mask_list = []
       
        for item in mask_token_list:
            
            if item in self.w2ix_s:
                self
                mask_list.append(self.w2ix_s[item])
                
            else:
                
                pass
            
        masks = 0
              
        for item in mask_list:
        
            masks += (x == item)
    
        return masks

    
    def _filter_and_sort_(self, remove_zero_seq = True, sort_order = True):
        
        self.entity_mask_train = [self._entity_mask_(x) for x in self.x_train_pad]
        self.entity_mask_dev = [self._entity_mask_(x) for x in self.x_dev_pad]
        self.entity_mask_test = [self._entity_mask_(x) for x in self.x_test_pad]
      
        self.training_prebatch = list(zip(self.x_train_pad, self.train_lengths, self.y_train_pad, self.y_train_lengths, self.entity_mask_train ))
        self.dev_prebatch = list(zip(self.x_dev_pad, self.dev_lengths, self.y_dev_pad, self.y_dev_lengths, self.entity_mask_dev))
        self.testing_prebatch = list(zip(self.x_test_pad, self.test_lengths, self.y_test_pad, self.y_test_lengths, self.entity_mask_test ))
        
        if sort_order:
        
            self.testing_prebatch = sorted(self.testing_prebatch, key = lambda x : x[3], reverse = True)
            self.training_prebatch = sorted(self.training_prebatch, key = lambda x : x[3], reverse = True)
            self.dev_prebatch = sorted(self.dev_prebatch, key = lambda x : x[3], reverse = True)
        
        if remove_zero_seq:
            self.training_prebatch = [x for x in self.training_prebatch if x[1] > 2][:50]
            self.dev_prebatch = [x for x in self.dev_prebatch if x[1] > 2][:10]
            self.testing_prebatch = [x for x in self.testing_prebatch if x[1] > 2][:10]
            
    def _batch_(self):
        
        B_SIZE = self._batch_size_
        
        training = DataLoader(self.training_prebatch, batch_size = B_SIZE, 
                          shuffle = False, pin_memory = False)
    
        development = DataLoader(self.dev_prebatch, batch_size = B_SIZE, 
                       shuffle = False, pin_memory = False)


        testing = DataLoader(self.testing_prebatch, batch_size = B_SIZE, 
                       shuffle = False, pin_memory = False)
        
        return training, development, testing
    
    def return_iterators(self, mask_token_list = ["<PAD>"]):
        
        self._pad_()
        self._to_tensor_
        self._filter_and_sort_()
        return self._batch_()
    
    
