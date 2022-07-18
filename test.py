from dataloader import my_dataset
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
class inference():
    def __init__(self):
        'Load the dataset and labels'
        self.dataset = my_dataset()
        self.labels = self.dataset.get_labels()

    def predict(self,text):
        'Load the pre-trained model and compile it'
        model= load_model("models/simple_network")
        model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        'Tokinzation and encoding of the input text'
        tokenizer = Tokenizer(num_words = 5000)
        tokenizer.fit_on_texts(self.dataset.data.text.values)
        tw = tokenizer.texts_to_sequences([text])
        tw = pad_sequences(tw, maxlen = 200)
        'Make prediction'
        prediction = int(model.predict(tw).round().item())
        print(model.predict(tw))
        'Return the prediction'
        return (self.labels)[1][prediction]