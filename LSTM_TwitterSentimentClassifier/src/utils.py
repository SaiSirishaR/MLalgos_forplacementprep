import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

## Pre-processing steps

class Preprocessing:
      
      def __init__(self, max_len, max_num_words):  
        
          self.data = 'data/tweets.csv'
          self.max_num_words = max_num_words
          self.max_len = max_len
            
      def load_data(self):
        
          data_inp = pd.read_csv(self.data)
          data_inp = data_inp.drop(['location','keyword', 'id'], axis = 1)     
          text_x = data_inp['text'].values         
          labels_y = data_inp['target'].values            
          self.train_text_x, self.test_text_x, self.train_labels_y, self.test_labels_y = train_test_split(text_x, labels_y, test_size=0.33)
          print("train size", np.shape(self.train_text_x), "test shape", np.shape(self.test_text_x))  
   
      def prepare_tokens(self):
          self.tokeniser = Tokenizer(num_words = self.max_num_words)
          self.fit_data = self.tokeniser.fit_on_texts(self.train_text_x)
            
      def sequence_to_token(self, xtexts): 
          sequences = self.tokeniser.texts_to_sequences(xtexts)
          return sequence.pad_sequences(sequences, maxlen=self.max_len)
        

        
