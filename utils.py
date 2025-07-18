import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Define the path where your model is stored
MODEL_PATH = 'model/your_model_file.h5' # <--- IMPORTANT: Update this with your actual model file name

# Load the model only once when the app starts
@st.cache_resource # Use st.cache_resource for heavy objects like models
def load_melanoma_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

def preprocess_image(image_file):
    """
    Preprocesses the uploaded image for the model.
    Assumes your model expects images of a specific size (e.g., 224x224) and normalized.
    """
    img = Image.open(image_file).convert('RGB') # Ensure image is in RGB format
    img = img.resize((224, 224)) # Resize to your model's expected input size
    img_array = np.asarray(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = img_array / 255.0 # Normalize pixel values to [0, 1]
    return img_array

def predict_melanoma(image_file, model):
    """
    Makes a prediction using the loaded model.
    """
    processed_image = preprocess_image(image_file)
    predictions = model.predict(processed_image)

    # Assuming a binary classification (melanoma/not melanoma)
    # Adjust this part based on your model's output (e.g., softmax for multiple classes)
    melanoma_probability = predictions[0][0] # Assuming 0 is benign, 1 is melanoma or vice versa

    # You might have different class names depending on your training
    class_names = ["Benign", "Malignant"] # Example class names
    
    # Determine the predicted class
    if melanoma_probability > 0.5:
        predicted_class = "Malignant (Melanoma)"
        confidence = melanoma_probability * 100
    else:
        predicted_class = "Benign (Non-Melanoma)"
        confidence = (1 - melanoma_probability) * 100 # Confidence for the benign class

    return predicted_class, confidence, melanoma_probability