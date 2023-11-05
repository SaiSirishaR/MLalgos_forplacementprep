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


### Prepare input to Dataloader

class SentimentDataset(Dataset):
    
      def __init__(self, x, y):
          
          self.x = x
          self.y = y  
            
      def __len__(self):
          return len(self.x)
        
      def __getitem__(self, idx):
          return self.x[idx], self.y[idx]
        
        
### Calculating the accuracy

def calculate_accuray(grand_truth, predictions):
    true_positives = 0
    true_negatives = 0

    for true, pred in zip(grand_truth, predictions):
        if (pred > 0.5) and (true == 1):
            true_positives += 1
        elif (pred < 0.5) and (true == 0):
             true_negatives += 1
        else:
             pass

    return (true_positives+true_negatives) / len(grand_truth)          

if __name__ == "__main__":
    
    ### Data initialisation and preprocessing

    data_initl= Preprocessing(max_len = 54, max_num_words = 1000)
    data_initl.load_data()
    data_initl.prepare_tokens()

    rtrain_x = data_initl.train_text_x
    rtest_x = data_initl.test_text_x

    train_y = data_initl.train_labels_y
    test_y = data_initl.test_labels_y

    train_x = data_initl.sequence_to_token(rtrain_x)
    test_x = data_initl.sequence_to_token(rtest_x)


    ## Hyperparameters
    
    bs = 32
    learning_rate = 0.01
    epochs = 5
    hidden_size = 128
    num_layers = 2
    embed_size= 16
    output_size=1
    input_size=1000
    
    #### Data prep for training
    
    train_set = SentimentDataset(train_x, train_y)
    test_set = SentimentDataset(test_x, test_y)
    
    dataloaded = DataLoader(train_set, batch_size=bs, shuffle=True)
    test_dataloaded = DataLoader(test_set)
    
    ### calling model, error function and optimisation
    model = LSTMclassifier(input_size, hidden_size, output_size, embed_size, num_layers)
    criterion = nn.BCELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    
    ### Training
    
    for epoch in range(epochs):
        
        train_prediction = []
        for x_inp, y_label in dataloaded:

           x = torch.tensor(x_inp, dtype=torch.long)
           y = torch.tensor(y_label, dtype=torch.float)
           pred_y = model(x)
           
           train_prediction += list(pred_y.squeeze().detach().numpy()) 
   
        ## loss calculation and optimisation
        
           loss = criterion(pred_y.view(-1), y)
           loss.backward()
           optimizer.step()
           optimizer.zero_grad() 
               
           
        train_accuracy = calculate_accuray(y_label, train_prediction) 
        print("Epoch ", epoch, "loss: ", loss.item(), "Train accuracy:", train_accuracy) 
           
  ### Testing the model
    test_prediction = []
    with torch.no_grad():
          for text_x, text_y in test_dataloaded:
                texts_x = torch.tensor(text_x, dtype=torch.long)
                labels_y = torch.tensor(text_y, dtype=torch.float)
                pred_test_y = model(texts_x)
                test_prediction += list(pred_test_y.detach().numpy())
    
                
    test_accuracy = calculate_accuray(text_y, test_prediction)

    print("Test accuracy: %.5f" % (test_accuracy))
   
