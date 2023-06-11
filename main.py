from fastapi import FastAPI, UploadFile, File
from model import read_image, preprocess, predict
import numpy as np

app = FastAPI()


@app.get('/')
def hello_world():
    return {'hello': 'world'}


@app.post('/predict')
async def predict_image(file: bytes = File(...)):
    image = read_image(file)
    image = preprocess(image)
    prediction = predict(image)

    return prediction

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
