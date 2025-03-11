import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import requests
import cv2
import numpy as np
import time

# ✅ Streamlit Branding
st.set_page_config(page_title="Live Face Recognition", page_icon="👀")

st.title("🔴 Live Face Recognition | CareerUpskillers")
st.write("🚀 Developed by [CareerUpskillers](https://www.careerupskillers.com)")
st.write("📞 Contact: WhatsApp 917975931377")

# ✅ Step 1: Download Model if Not Exists
MODEL_PATH = "models/face_recognition_model.pth"
MODEL_URL = "https://drive.google.com/uc?id=1hjMmDWjWpJK4ewXWmEAr4i5Aq6newfIC"  # Update with latest Google Drive link

if not os.path.exists(MODEL_PATH):
    os.makedirs("models", exist_ok=True)
    st.warning("📥 Downloading model file... Please wait.")
    try:
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        st.success("✅ Model Downloaded Successfully!")
        time.sleep(2)
        st.rerun()
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")
        st.stop()

# ✅ Step 2: Define Face Recognition Model
class FaceRecognitionModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognitionModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(64 * 16 * 16, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ✅ Step 3: Dynamically Count Classes in Dataset
DATASET_PATH = "data/train"
if os.path.exists(DATASET_PATH):
    classes = os.listdir(DATASET_PATH)
else:
    classes = ["Unknown"]  # Default class

num_classes = len(classes)  # Match model structure

# ✅ Step 4: Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FaceRecognitionModel(num_classes=num_classes).to(device)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    st.success("✅ Model Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ✅ Step 5: Define Image Processing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ✅ Step 6: Streamlit UI for Live Face Recognition
st.write("## 🎥 Live Face Recognition")

col1, col2 = st.columns(2)
with col1:
    start_btn = st.button("▶️ Start Live Face Recognition")
with col2:
    stop_btn = st.button("⏹️ Stop Live Face Recognition")

if "run" not in st.session_state:
    st.session_state.run = False

if start_btn:
    st.session_state.run = True
    st.rerun()

if stop_btn:
    st.session_state.run = False
    st.rerun()

# ✅ Step 7: Live Webcam Feed
if st.session_state.run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Unable to access webcam!")
        st.session_state.run = False
    else:
        st.success("📷 Webcam Activated! Press 'Stop' to close.")
        frame_placeholder = st.empty()

        while st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to capture frame from webcam.")
                break

            # Convert OpenCV frame to PIL
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tensor = transform(img).unsqueeze(0).to(device)

            # Predict Class
            with torch.no_grad():
                output = model(img_tensor)
                predicted_class = torch.argmax(output).item()

            label = classes[predicted_class] if predicted_class < len(classes) else "Unknown"

            # Display Result on Screen
            cv2.putText(frame, f"Prediction: {label}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame in Streamlit
            frame_placeholder.image(frame, channels="BGR", use_column_width=True)

            time.sleep(0.1)

        cap.release()
        cv2.destroyAllWindows()
        st.warning("🔴 Webcam Closed!")
