from tensorflow import keras
from io import BytesIO
import numpy as np
from fastapi import FastAPI, File
from fastapi.responses import JSONResponse
import uvicorn

model = keras.models.load_model('ModelML.h5', compile=False)
model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['accuracy'])
img_size = (224, 224)
class_names = ['ayam_bakar', 'bakso', 'gado_gado', 'rendang', 'sate']


def load_image(uploaded_image):
    # Load and resize the image using Keras preprocessing
    img = keras.preprocessing.image.load_img(
        BytesIO(uploaded_image), target_size=img_size)
    return img


def preprocessing(img):
    # Convert the image to an array
    img_array = keras.preprocessing.image.img_to_array(img)

    # Reshape the array to match the input shape of the model
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize the image data
    img_array = img_array / 255.0
    return img_array


def predict_image(img_array):
    prediction = model.predict(img_array)
    print(prediction)

    # Get the predicted class label
    predicted_class_idx = np.argmax(prediction, axis=-1)[0]
    predicted_class_label = class_names[predicted_class_idx]
    return predicted_class_label
