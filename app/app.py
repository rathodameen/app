import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import streamlit as st
import os

# Load the model and feature extractor
class PneumoniaModel:
    def __init__(self):
        self.model_name = "google/vit-base-patch16-224-in21k"  # Pretrained ViT model
        self.model = ViTForImageClassification.from_pretrained(self.model_name)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.model_name)

    def predict(self, image_path):
        # Open the image
        image = Image.open(image_path)

        # Preprocess the image
        inputs = self.feature_extractor(images=image, return_tensors="pt")

        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        return predictions.item()

# Instantiate the model
model = PneumoniaModel()

# Streamlit app
st.title("Pneumonia Detection Web App")
st.write("Upload a chest X-ray image to check for pneumonia.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    image_path = "temp_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    image = Image.open(image_path)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make prediction
    prediction = model.predict(image_path)

    # Display result
    if prediction == 0:
        st.write("**The image shows a healthy lung.**")
    else:
        st.write("**The image shows signs of pneumonia.**")

    # Remove the temporary image file
    os.remove(image_path)
