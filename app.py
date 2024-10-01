import streamlit as st
import numpy as np
import cv2
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils import normalize
from PIL import Image
import matplotlib.pyplot as plt

# Load the model
@st.cache(allow_output_mutation=True)
def load_brain_tumor_model():
    model = load_model('brain_tumor.hdf5')
    return model

model = load_brain_tumor_model()

# Define function to preprocess the image
def preprocess_image(image, img_size=256):
    image = np.array(image.convert('L'))  # Convert image to grayscale if not already
    image = cv2.resize(image, (img_size, img_size))  # Resize the image to model input size
    image = normalize(np.array(image), axis=1)  # Normalize the image
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)   # Add batch dimension
    return image

# Streamlit app
st.title("Brain Tumor Segmentation")

st.write("""
         This app uses a U-Net model trained to perform brain tumor segmentation.
         Upload an MRI scan, and the app will highlight the predicted tumor area.
         """)

# Upload image
uploaded_file = st.file_uploader("Upload an MRI scan image", type=["jpg", "png", "jpeg", "tif"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI scan.', use_column_width=True)
    
    # Preprocess the image for the model
    preprocessed_image = preprocess_image(image)
    
    # Make prediction
    st.write("Classifying...")
    prediction = (model.predict(preprocessed_image)[0, :, :, 0] > 0.2).astype(np.uint8)
    
    # Display the prediction
    st.write("Prediction (Tumor Segmentation):")
    
    # Convert prediction back to image format for display
    pred_image = Image.fromarray((prediction * 255).astype(np.uint8))
    st.image(pred_image, caption="Predicted Tumor Area", use_column_width=True)

    # Optionally, display both original and prediction side by side
    st.write("Original vs Prediction:")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(np.array(image.convert('L')), cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    
    ax[1].imshow(prediction, cmap='gray')
    ax[1].set_title('Predicted Tumor Area')
    ax[1].axis('off')
    
    st.pyplot(fig)