import os

def get_file_size(file_path):
    """Returns the size of file in bytes"""
    return os.path.getsize(file_path)

# Example usage
file_path = r"C:\Users\VINIL\Desktop\soil_type_pred_ML\flask_app\models\soil_cnn_model.keras"

size_bytes = get_file_size(file_path)
print(f"Size of {file_path}: {size_bytes} bytes")
print(f"Size in KB: {size_bytes / 1024:.2f} KB")
print(f"Size in MB: {size_bytes / (1024 * 1024):.2f} MB")