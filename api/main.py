from io import BytesIO
import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image

app = FastAPI()

MODEL = tf.keras.models.load_model("../training/saved_models/1")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello I am Alive"


# def read_file_as_image(data) -> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).resize((256, 256))
    image_array = np.array(image)
    return image_array


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(image_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence':float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)