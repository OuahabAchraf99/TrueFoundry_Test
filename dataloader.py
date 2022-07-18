import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class my_dataset():
    'Generates data'
    def __init__(self):
        'Loading data'
        self.data = pd.read_csv(r"airline_sentiment_analysis.csv")
        
    def preprocess(self):
        'Preprocessing data'
        reviews=self.data.text.values 

        'Tokenize the text'
        tokenizer = Tokenizer(num_words = 5000)
        tokenizer.fit_on_texts(reviews)
        vocab_size = len(tokenizer.word_index) + 1

        'Encode the text'
        encoded_doc = tokenizer.texts_to_sequences(reviews)
        
        padded_sequence = pad_sequences(encoded_doc, maxlen = 200)

        return padded_sequence, vocab_size

    def get_labels(self):
        'Return a converted labels positive/negative to 0/1'
        return self.data.airline_sentiment.factorize()

