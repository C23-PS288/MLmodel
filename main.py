from fastapi import FastAPI, UploadFile, File
from model import read_image, preprocess, predict
import numpy as np

app = FastAPI()


@app.get('/')
def hello_world():
    return {'hello': 'world'}


@app.post('/predict')
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = read_image(np.frombuffer(contents, np.uint8))
    image = preprocess(image)
    prediction = predict(image)

    return prediction

# if __name__ == "__main__":
#     uvicorn.run(app, port=8080, host='0.0.0.0')


# model = k.models.load_model('ModelML.h5', compile=False)
# model.compile(optimizer='adam', loss='categorical_crossentropy',
#               metrics=['accuracy'])
# img_size = (224, 224)
# class_names = ['ayam_bakar', 'bakso', 'gado_gado', 'rendang', 'sate']


# @app.post('/predict')
# async def predicted_image(*, file: UploadFile = File(...)):
#     # Load the image
#     img = k.preprocessing.image.load_img(file.file, target_size=img_size)

#     # Convert the image to an array
#     img_array = k.preprocessing.image.img_to_array(img)

#     # Reshape the array to match the input shape of the model
#     img_array = np.expand_dims(img_array, axis=0)

#     # Normalize the image data
#     img_array = img_array / 255.0

#     # Make a prediction using the trained model
#     prediction = model.predict(img_array)

#     # Get the predicted class label
#     predicted_class_idx = np.argmax(prediction, axis=-1)[0]
#     predicted_class_label = class_names[predicted_class_idx]
#     return {"prediction": predicted_class_label}
