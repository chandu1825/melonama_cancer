import streamlit as st
from utils import load_melanoma_model, predict_melanoma
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Melanoma Detector",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("🔬 Melanoma Detector")
st.markdown("Upload an image of a skin lesion to get a prediction on whether it's benign or malignant (melanoma).")
st.markdown("---")

# Load the model
try:
    model = load_melanoma_model()
    st.success("Deep Learning Model Loaded Successfully!")
except FileNotFoundError as e:
    st.error(str(e))
    st.info("Please ensure your trained model file (`your_model_file.h5`) is in the `model/` directory.")
    st.stop() # Stop the app if the model isn't found
except Exception as e:
    st.error(f"Failed to load model. Please check `utils.py` and your model file. Error: {e}")
    st.stop()


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Make prediction
    predicted_class, confidence, raw_probability = predict_melanoma(uploaded_file, model)

    st.markdown("---")
    st.subheader("Prediction Results:")

    if "Malignant" in predicted_class:
        st.error(f"**Predicted Class: {predicted_class}**")
        st.warning(f"Confidence: {confidence:.2f}%")
        st.write("It is advisable to consult a dermatologist for further examination if the prediction indicates a potential malignancy.")
    else:
        st.success(f"**Predicted Class: {predicted_class}**")
        st.info(f"Confidence: {confidence:.2f}%")
        st.write("While the model predicts this lesion as benign, it's always good practice to monitor any suspicious skin changes and consult a healthcare professional if you have concerns.")

    st.markdown("---")
    st.write(f"Raw Model Output (Probability of Malignant): {raw_probability:.4f}")

st.markdown("---")
st.caption("Disclaimer: This tool is for informational purposes only and should not be used as a substitute for professional medical advice. Always consult with a qualified healthcare provider for any health concerns.")