from tensorflow import keras as k
from fastapi import FastAPI, File
from fastapi.responses import JSONResponse
import uvicorn
from models import load_image, preprocessing, predict_image

app = FastAPI()


@app.post('/predict')
async def GetFoodName(imgpath: bytes = File(...)):
    img = load_image(imgpath)
    img_array = preprocessing(img)
    predicted_class_label = predict_image(img_array)
    return JSONResponse(content={"predicted_class_label": predicted_class_label})

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
