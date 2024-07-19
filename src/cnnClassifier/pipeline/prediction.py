import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load the model
        model_path = os.path.join("artifacts", "training", "model.h5")
        model = load_model(model_path)
        
        # Load and preprocess the image
        try:
            img = image.load_img(self.filename, target_size=(224, 224))
        except Exception as e:
            print(f"Error loading image: {e}")
            return [{"image": "Error loading image"}]
        
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0

        # Make predictions
        try:
            predictions = model.predict(img_array)
            result = np.argmax(predictions, axis=1)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return [{"image": "Error during prediction"}]
        
        # Interpret the result
        if result[0] == 1:
            prediction = "Normal"
        else:
            prediction = "Adenocarcinoma"
        
        return [{"image": prediction}]
