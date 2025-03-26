import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import logging

# 🔹 Setup Logging
logging.basicConfig(level=logging.INFO)
base_dir = os.path.dirname(os.path.abspath(__file__))
cnn_model_path = os.path.join(base_dir,"models","soil_cnn_model.keras")
class_indices_path = os.path.join(base_dir,"models","class_indices.pkl")

# 🔹 Load the Saved CNN Model
logging.info(f"Loading CNN model from: {cnn_model_path}")
model = tf.keras.models.load_model(cnn_model_path)
logging.info("CNN model loaded successfully.")

# 🔹 Load Class Indices
logging.info(f"Loading class indices from: {class_indices_path}")
with open(class_indices_path, "rb") as f:
    class_indices = pickle.load(f)

# Reverse the class indices dictionary {class_name: index} --> {index: class_name}
index_to_class = {v: k for k, v in class_indices.items()}
logging.info(f"Class index mapping: {index_to_class}")

# 🔹 Function to Predict Soil Type from a Single Image
def predict_soil_type(img_path):
    try:
        logging.info(f"Predicting soil type for image: {img_path}")

        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(img_array)
        
        predicted_index = np.argmax(predictions, axis=1)[0]
        predicted_class = index_to_class[predicted_index]
        confidence = predictions[0][predicted_index]

        logging.info(f"Predicted class index: {predicted_index}")
        logging.info(f"Predicted soil type: {predicted_class} with confidence {confidence:.2f}")

        print(f"Predicted Soil Type: {predicted_class}")
        print(f"Confidence: {confidence * 100:.2f}%")

        return {
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "predictions": predictions.tolist()  # Return raw prediction scores if needed
        }

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return {"error": str(e)}

# 🔹 Example Usage
if __name__ == "__main__":
    # Provide the path to your test image here
    test_image_path = r"C:\Users\VINIL\Desktop\soil_type_ML\test_images\sample_soil.jpg"

    result = predict_soil_type(test_image_path)
    print(result)
