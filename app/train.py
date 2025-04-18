import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import numpy as np


class PneumoniaModel:
    def __init__(self):
        self.model_name = "google/vit-base-patch16-224-in21k"  # Pretrained ViT model
        self.model = ViTForImageClassification.from_pretrained(self.model_name)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.model_name)

    def predict(self, image_path):
        # Load the image
        image = Image.open(image_path)

        # Preprocess the image
        inputs = self.feature_extractor(images=image, return_tensors="pt")

        # Make the prediction
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        return predictions.item()


# Instantiate the model
model = PneumoniaModel()
