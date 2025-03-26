from flask import Blueprint, request, jsonify
import logging
import traceback
import os
from werkzeug.utils import secure_filename

# Import the CNN prediction function from ml_utils.py
from flask_app.ml_utils import predict_soil_type

# Setup logging
log_file = "notebooks/api.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Create Blueprint
routes = Blueprint('routes', __name__)

# Folder to temporarily store uploaded images
UPLOAD_FOLDER = "flask_app/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@routes.route('/predict-soil', methods=['POST'])  # Endpoint matches your frontend fetch call
def predict_soil_api():
    try:
        # Check if the file was uploaded correctly
        if 'soilImage' not in request.files:
            logging.warning("No file with key 'soilImage' found in request.")
            return jsonify({
                "success": False,
                "error": "No image provided. Please upload an image with key 'soilImage'."
            }), 400

        file = request.files['soilImage']

        if file.filename == '':
            logging.warning("Empty filename received.")
            return jsonify({
                "success": False,
                "error": "No file selected. Please upload a valid image."
            }), 400

        # Secure and save the uploaded file temporarily
        filename = secure_filename(file.filename)
        img_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(img_path)
        logging.info(f"Image successfully uploaded to: {img_path}")

        # Run the prediction using your CNN model
        result = predict_soil_type(img_path)

        # Delete the image after prediction (cleanup)
        os.remove(img_path)
        logging.info(f"Temporary file deleted: {img_path}")

        # Return the prediction
        return jsonify({
            "success": True,
            "prediction": result.get("predicted_class"),
            "confidence": f"{result.get('confidence') * 100:.2f}%",
            "raw_predictions": result.get("predictions")  # Optional, you can remove if not needed
        })

    except Exception as e:
        error_message = traceback.format_exc()
        logging.error(f"POST /predict-soil - Error:\n{error_message}")

        return jsonify({
            "success": False,
            "error": "Prediction failed due to server error."
        }), 500
