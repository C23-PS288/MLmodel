from PIL import Image
from io import BytesIO
import numpy as np
from tensorflow import keras as k

input_shape = (224, 224)
class_names = ['ayam_bakar', 'bakso', 'gado_gado', 'rendang', 'sate']


def load_model():
    model = k.models.load_model('ModelML.h5', compile=False)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = load_model()


def predict(image_array: np.ndarray):
    # Normalize the image data
    img_array = image_array / 255.0

    # Make a prediction using the trained model
    prediction = model.predict(img_array)
    print(prediction)

    # Get the predicted class label
    predicted_class_idx = np.argmax(prediction, axis=-1)[0]
    predicted_class_label = class_names[predicted_class_idx]
    return predicted_class_label


def read_image(image_encoded):
    pil_image = Image.open(BytesIO(image_encoded))
    return pil_image


def preprocess(image: Image.Image):
    image = image.resize(input_shape)
    image = np.asfarray(image)
    image = image / 255.0
    image = np.expand_dims(image, 0)
    return image
