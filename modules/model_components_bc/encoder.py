import torch
import torch.nn as nn
import math 
from transformers import AutoModel, AutoConfig
import json

with open('modules/config.txt', 'r') as f:
    args = json.load(f)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, 
                 encode_sel = "lstm", layers=1, dropout=0., 
                 nmt = None, bidirectional=True, embedding = None):
        
        super(Encoder, self).__init__()
        
        self.bidirectional = bidirectional

        self.dropout = dropout
        self.encode_sel = encode_sel
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
 
        self.vocab_size = vocab_size

        if embedding is not None:
            
            self.embedding = nn.Embedding(vocab_size, 
                                    embedding_dim, 
                                    padding_idx=0)
            
            self.embedding.load_state_dict({"weight":torch.tensor(embedding).float()})
            
            
            ## assertion that pretrained embeddings where loaded correctly
            assert torch.eq(self.embedding.weight, torch.tensor(embedding).float()).sum() == self.embedding.weight.numel()        
            
            # we do not train pretrained embeddings
            self.embedding.weight.requires_grad = False
            
        else:
            
            if encode_sel == "bert":
                
                pass
            
            else:
                
                self.embedding = nn.Embedding(vocab_size, 
                                        embedding_dim, 
                                        padding_idx=0)
                self.embedding.weight.data.uniform_(0.0,1.0)
            

        if encode_sel == "mlp":
           
            self.average = nn.Linear(embedding_dim, hidden_dim*2)
            stdv = 1. / math.sqrt(embedding_dim)
            self.average.weight.data.uniform_(-stdv,stdv)
            self.average.bias.data.fill_(0)
            

        elif encode_sel == "cnn":
            
            self.activation = nn.ReLU()
            
            kernel_size = [1,3,5,7]
            convs = {}
            for i in range(len(kernel_size)) :
                convs[str(i)] = nn.Conv1d(embedding_dim, hidden_dim * 2// len(kernel_size), kernel_size[i], 
                                          padding=int((kernel_size[i] - 1)//2))
    
            self.convolutions = nn.ModuleDict(convs)
            
        elif encode_sel == "bert":
            
            self.bert_config = AutoConfig.from_pretrained(args["bert_model"])   
            self.bert = AutoModel.from_pretrained(args["bert_model"], config=self.bert_config)

            self.embedding_dim = 768
            
        elif (encode_sel == "lstm") or (encode_sel == "gru"):
		
	        encode_sel = getattr(torch.nn, encode_sel.upper())

	        self.dropout = nn.Dropout(p = self.dropout)
          
                
	        self.rnn = encode_sel(embedding_dim, 
	                           hidden_dim, 
	                           layers, 
	                           dropout=dropout, 
	                           bidirectional=bidirectional,
                               batch_first = True)
            
            
        
    def forward(self, input, lengths, hidden=None, ig = int(1)):

        if self.encode_sel == "bert":
            
            inpt_seq = input.long()
            atten_mask = input != 0

            if len(inpt_seq[0]) == 0:
                
                output = torch.zeros([len(inpt_seq), 1, self.embedding_dim]).to(device)
                last_hidden = torch.zeros([len(inpt_seq), self.embedding_dim]).to(device)
        
            else:
                
                ## bert decomposed for obtaining ig and embed grads

                if inpt_seq is not None:
                    input_shape = inpt_seq.size()

                seq_length = input_shape[1]
                position_ids = torch.arange(512).expand((1, -1)).to(device)
                position_ids = position_ids[:, :seq_length]
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=position_ids.device)


                self.embed = self.bert.embeddings.word_embeddings(inpt_seq)
                position_embeddings = self.bert.embeddings.position_embeddings(position_ids)
                token_type_embeddings = self.bert.embeddings.token_type_embeddings(token_type_ids)

                embeddings = self.embed + position_embeddings + token_type_embeddings
                embeddings = self.bert.embeddings.LayerNorm(embeddings)
                embeddings = self.bert.embeddings.dropout(embeddings)

                extended_attention_mask = atten_mask.unsqueeze(1).unsqueeze(2)

                extended_attention_mask = extended_attention_mask.to(dtype=next(self.bert.parameters()).dtype) # fp16 compatibility
                extended_attention_mask = (1 - extended_attention_mask) * -10000.0

                head_mask = [None] * self.bert.config.num_hidden_layers

                encoder_outputs = self.bert.encoder(embeddings * ig, extended_attention_mask, head_mask=head_mask)[0][:,:max(lengths)]

                output = encoder_outputs
                last_hidden = self.bert.pooler(output)
  
        elif self.encode_sel == "mlp":

            self.embed = self.embedding(input.long()) 

            embed_ig = self.embed * ig

            output = nn.Tanh()(self.average(embed_ig[:,:max(lengths),:]))
            
            last_hidden = output.mean(1)
            
        elif self.encode_sel == "cnn":
            
            self.embed = self.embedding(input.long())

            embed_ig = self.embed * ig
            
            seq_t = embed_ig.transpose(1,2)
             
            outputs = [self.convolutions[i](seq_t) for i in sorted(self.convolutions.keys())]
            
            outputs = self.activation(torch.cat(outputs, dim = 1))
    
            last_hidden = nn.functional.max_pool1d(outputs, kernel_size=outputs.size(-1)).squeeze(-1)
            
            output = outputs.transpose(1,2)[:, :int(max(lengths)),:]
                      
        else:
            
            self.embed = self.embedding(input.long())

            embed_ig = self.embed * ig
   
            packseq = nn.utils.rnn.pack_padded_sequence(embed_ig, lengths, batch_first=True, enforce_sorted = False)
            
            output, (h,c) = self.rnn(packseq)

            output, lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, padding_value=0)
                
            output = self.dropout(output)
            
            if self.encode_sel == "lstm":
            
                last_hidden = torch.cat([h[0], h[1]], dim = -1)
                
            else:
                
                last_hidden = torch.cat([h, c], dim = -1)
    
        return output, last_hidden

