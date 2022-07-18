# main.py
from model import my_model
from test import inference
from fastapi import FastAPI

app = FastAPI()
model=my_model()
inference=inference()
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/train")
async def train_model():
    ''' This route build a model and launch the training process. Trained Model is saved under the name simple_network in the path './models/simple_network' .'''
    model.build_model()
    model.train()
    return {"Model trained successfully !"}

@app.get("/predict/{text}")
async def predict_sentiment(text:str):
    ''' This route makes an inference on an english text and respond with the predicted sentiment. The saved model is used to make predictions.'''
    prediction=inference.predict(str(text))
    return {"Predicted Sentiment: ": prediction}
