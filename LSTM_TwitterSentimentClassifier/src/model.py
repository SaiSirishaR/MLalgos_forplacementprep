### Adding evaluation

####### Coding LSTM for Twitter classification


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

## importing torch modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


##### LSTM class

class LSTMclassifier(nn.Module):
    
      def __init__(self, input_size, hidden_size, output_size, embed_size, num_layers):

            super(LSTMclassifier, self).__init__()
            
            self.num_layers = num_layers
            self.embed_size = embed_size
            self.hidden_size = hidden_size
            
            
            self.embedding = nn.Embedding(input_size, embed_size)
            self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first =True)
            self.fc1 = nn.Linear(hidden_size, output_size)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(output_size,1)
            
      def forward(self, inp): 
        
           ### Initializing hidden layers
            
           h0 = torch.zeros((self.num_layers, inp.size(0), self.hidden_size)) # hidden state
           c0 = torch.zeros((self.num_layers, inp.size(0), self.hidden_size)) # cell state
        
           #torch.nn.init.xavier_normal_(h)
           #torch.nn.init.xavier_normal_(c)
        
           embedded = self.embedding(inp)
           out, (ht,ct) = self.lstm(embedded, (h0,c0)) 
           out = self.fc1(out[:,-1,:])
           out = self.dropout(out) 
           out = torch.sigmoid(self.fc2(out))
           return out
    
   
