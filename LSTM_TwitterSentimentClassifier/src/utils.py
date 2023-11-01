import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

class Preprocessing:
      
      def __init__(self):  
          self.data = 'data/tweets.csv'
            
      def load_data(self):
          data_inp = pd.read_csv(self.data)
          data_inp = data_inp.drop(['location','keyword', 'id'], axis = 1)   
          text_x = data_inp['text'].values    
          labels_y = data_inp['target'].values  
          self.train_text_x, self.train_labels_y, self.test_text_x, self.test_labels_y = train_test_split(text_x, labels_y, test_size=0.33)
          
   
      def prepare_tokens(self):
          self.tokeniser = Tokenizer(num_words = max_num_words)
          self.fit_data = self.tokeniser.fit_on_texts(self.train_text_x)
            
      def sequence_to_token(self, xtexts): 
          sequences = self.tokenizer.texts_to_sequences(stexts)
          return sequence.pad_sequences(sequences, maxlen=self.max_len)
        
