import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from transformers import ViTForImageClassification, AutoImageProcessor

# ===== CACHED: Load model and processor only once =====
@st.cache_resource
def load_model_and_processor():
    model_name = "nickmuchi/vit-finetuned-chest-xray-pneumonia"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name, output_attentions=True)
    return model, processor

model, processor = load_model_and_processor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ===== Convert grayscale to RGB if needed =====
def convert_to_rgb(image):
    return image.convert("RGB") if image.mode != "RGB" else image

# ===== Apply attention visualization =====
def apply_attention_visualization(image, model, prediction_idx):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor, output_attentions=True)

    attentions = outputs.attentions[-1].squeeze(0).mean(dim=0)
    attention_map = cv2.resize(attentions.mean(dim=0).cpu().numpy(), image.size)

    attention_map = np.clip((attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8), 0, 1)
    image_np = np.array(image)

    if prediction_idx == 1:
        threshold = np.percentile(attention_map, 98)
        mask = np.uint8(attention_map >= threshold)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        red_overlay = np.zeros_like(image_np)
        red_overlay[:, :, 0] = 255
        masked = cv2.bitwise_and(red_overlay, red_overlay, mask=mask)
        output = cv2.addWeighted(image_np, 0.7, masked, 0.3, 0)
    else:
        heatmap = np.uint8(255 * attention_map)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)
        output = cv2.addWeighted(image_np, 0.7, heatmap_color, 0.3, 0)

    return output

# ===== Predict pneumonia =====
def predict(image):
    image = convert_to_rgb(image)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.argmax(outputs.logits).item()

# ===== Streamlit App =====
def main():
    st.title("Pneumonia Detection from Chest X-ray")
    st.write("Upload a chest X-ray image to detect pneumonia.")

    uploaded_file = st.file_uploader("Upload X-ray image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((512, 512))  # Resize early to reduce memory
        st.image(image, caption="Uploaded X-ray", use_column_width=True)

        if st.button("Detect Pneumonia"):
            with st.spinner("Analyzing..."):
                prediction_idx = predict(image)

                if prediction_idx == 1:
                    st.error("🩺 Pneumonia Detected")
                else:
                    st.success("✅ No Pneumonia Detected")

                vis = apply_attention_visualization(image, model, prediction_idx)
                st.image(vis, caption="Attention Map", use_column_width=True)

if __name__ == "__main__":
    main()
