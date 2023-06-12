from fastapi import FastAPI, File
from fastapi.responses import JSONResponse
import uvicorn
from models import load_image, preprocessing, predict_image

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "connection success"}

@app.post('/predict')
async def GetFoodName(imgpath: bytes = File(...)):
    img = load_image(imgpath)
    img_array = preprocessing(img)
    predicted_class_label = predict_image(img_array)
    return JSONResponse(content={"predicted_class_label": predicted_class_label})


if __name__ == "__main__":
    port = 8001
    uvicorn.run(app, host="0.0.0.0",port=port)
