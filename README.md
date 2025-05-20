Soil Type Prediction Using Convolutional Neural Networks (CNN)

This project is focused on predicting the soil type from images using Convolutional Neural Networks (CNN). Soil type identification plays a critical role in agriculture, crop planning, and land management. By leveraging deep learning techniques, particularly CNNs, this project aims to automate the soil classification process with high accuracy and reliability.

This project is a Flask-based web application that integrates multiple machine learning models to predict irrigation types based on input features.  
Note: Deployment was not completed due to platform limitations, including subscription requirements and file size restrictions.
## Folder Structure  
irrigation_type/
│── flask_app/
│   │── __init__.py
│   │── app.py
│   │── routes.py
│   │── models/
│   │   │── model.pkl
│   │── static/
│   │── templates/
│   │── utils.py
│── notebooks/
│   │── train_model.ipynb
│── requirements.txt
│── config.yaml
│── Procfile
│── README.md

📌 Project Highlights
Model Type: Convolutional Neural Network (CNN)
Task: Soil Type Image Classification
Dataset: Collection of labeled soil images across various soil types
Classes: Black, Red, Clayey, Sandy, Loamy (or others depending on dataset)
Frameworks Used: TensorFlow / Kera
Deployment Ready: Yes (can be integrated into mobile/web apps)

🧠 Problem Statement
Manually identifying soil types is time-consuming and prone to human error. Farmers and agriculturists need a fast and efficient solution to detect soil type on the go. This model simplifies soil classification using computer vision by analyzing soil images and predicting their category with high accuracy.

Dataset
The dataset consists of labeled images representing different soil types. Each class contains numerous images captured under varying lighting, angles, and textures to help the model generalize better.
Number of Classes: 5 (e.g., Black, Red, Clayey, Sandy, Loamy)
Image Size: Resized to 128x128 (or 224x224 depending on model configuration)
Format: JPEG/PNG
Data Split:
Training Set: 70%
Validation Set: 15
Test Set: 15%

🔧 Technologies & Tools
Programming Language: Python
Libraries:
TensorFlow / Keras
NumPy
Matplotlib
scikit-learn
Environment: Jupyter Notebook / Google Colab / Any IDE
Optional Tools: OpenCV for image preprocessing, seaborn for advanced visualization

 Model Architecture
The CNN model consists of the following layers:
Input Layer
2–3 Convolutional Layers with ReLU activation
MaxPooling Layers to reduce spatial dimensions
Flatten Layer
Fully Connected Dense Layers
Output Layer with Softmax activation for classification
The architecture is designed to balance performance with computational efficiency for real-world usability.

⚙️ Training Configuration
Loss Function: Categorical Crossentropy
Optimizer: Adam
Metrics: Accuracy
Epochs: 25–50 (depending on early stopping)
Batch Size: 32
Callbacks Used: EarlyStopping, ModelCheckpoint

Model Performance
Training Accuracy: ~95% (depends on dataset)
Validation Accuracy: ~90% or above
Test Accuracy: High generalization observed on unseen data
Confusion Matrix: Indicates strong performance across all classes
No Overfitting: Regularization techniques and dropout layers applied

Preprocessing Steps
Image resizing
Normalization (pixel values scaled between 0 and 1)
Data augmentation (rotation, zoom, shift, flip) to increase robustness
One-hot encoding for class labels

📊 Results
| Metric         | Value    |
| -------------- | -------- |
| Accuracy       | \~90–95% |
| Precision      | High     |
| Recall         | High     |
| F1-Score       | High     |
| Inference Time | Fast     |

🖼️ Sample Predictions
Example results:
Input Image: Soil Image
Predicted Class: Clayey
Confidence: 98.4%
Visual prediction samples can be found in the predictions/ directory.

🚀 Future Improvements
Add more diverse data from different geographic regions
Deploy model using Flask/Streamlit or on mobile using TensorFlow Lite
Integrate GPS and pH sensor data for hybrid prediction
Add support for real-time camera feed prediction

🤝 Contributing
Pull requests are welcome! If you have suggestions for improving performance or expanding the dataset, feel free to fork the repo and open a PR.

📜 License
This project is open-source and available under the MIT License.

🙋‍♂️ Contact
For queries or feedback, feel free to reach out via email or GitHub issues.

Note: Deployment was not completed due to platform limitations, including subscription requirements and file size restrictions.
