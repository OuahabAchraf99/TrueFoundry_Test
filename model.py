from dataloader import my_dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, SpatialDropout1D
class my_model():

    def __init__(self):
        'load dataset and labels'
        data=my_dataset()
        self.padded_sequence,self.vocab_size=data.preprocess()
        self.labels= data.get_labels()

    def build_model(self):
        'Build the model'
        embedding_vector_length = 32
        vocab_size=self.vocab_size
        model = Sequential()
        model.add(Embedding(vocab_size, embedding_vector_length, input_length = 200))
        model.add(SpatialDropout1D(0.25))
        model.add(LSTM(50, dropout = 0.5, recurrent_dropout = 0.5))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation = 'sigmoid'))
        return model

    def train(self):
        'Create a model instance'
        model=self.build_model()
        'Compile the model and specify training parameters'
        model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        'Launch the training'
        history = model.fit(self.padded_sequence, self.labels[0], validation_split=0.2, epochs=100, batch_size=32)
        'Save the model'
        model.save("models/simple_network")
        return history