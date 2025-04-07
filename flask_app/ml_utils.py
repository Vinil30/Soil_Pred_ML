import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import logging

# 🔹 TensorFlow Memory Configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_virtual_device_configuration(
    tf.config.list_physical_devices('CPU')[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=400)]
)

# 🔹 Setup Logging
logging.basicConfig(level=logging.INFO)
base_dir = os.path.dirname(os.path.abspath(__file__))
cnn_model_path = os.path.join(base_dir, "models", "soil_cnn_model.keras")
class_indices_path = os.path.join(base_dir, "models", "class_indices.pkl")

# 🔹 Load the Saved CNN Model
logging.info(f"Loading CNN model from: {cnn_model_path}")
try:
    model = tf.keras.models.load_model(cnn_model_path)
except TypeError as e:
    logging.warning("Standard load failed, trying with custom objects...")
    model = tf.keras.models.load_model(
        cnn_model_path,
        custom_objects={'InputLayer': tf.keras.layers.InputLayer}
    )
logging.info("CNN model loaded successfully.")

# 🔹 Load Class Indices
logging.info(f"Loading class indices from: {class_indices_path}")
with open(class_indices_path, "rb") as f:
    class_indices = pickle.load(f)
index_to_class = {v: k for k, v in class_indices.items()}
logging.info(f"Class index mapping: {index_to_class}")

# 🔹 Prediction Function (Remains Same)
def predict_soil_type(img_path):
    try:
        logging.info(f"Predicting soil type for image: {img_path}")
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions, axis=1)[0]
        predicted_class = index_to_class[predicted_index]
        confidence = predictions[0][predicted_index]

        logging.info(f"Predicted: {predicted_class} (Confidence: {confidence:.2f})")
        return {
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "predictions": predictions.tolist()
        }
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    test_image_path = "test_images/sample_soil.jpg"
    print(predict_soil_type(test_image_path))
