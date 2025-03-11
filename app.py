import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import requests
import cv2
import numpy as np
import time
import tempfile

# ✅ Streamlit Branding
st.set_page_config(page_title="Live Face Recognition", page_icon="👀")

st.title("🔴 Live Face Recognition | CareerUpskillers")
st.write("🚀 Developed by [CareerUpskillers](https://www.careerupskillers.com)")
st.write("📞 Contact: WhatsApp 917975931377")

# ✅ Step 1: Download Model if Not Exists
MODEL_PATH = "models/face_recognition_model.pth"
MODEL_URL = "https://drive.google.com/uc?id=YOUR_NEW_MODEL_ID"  # Replace with correct ID

if not os.path.exists(MODEL_PATH):
    st.warning("📥 Downloading model file... Please wait.")
    response = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    st.success("✅ Model Downloaded Successfully!")
    time.sleep(2)
    st.experimental_rerun()

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

# ✅ Step 3: Load Model & Dataset Classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Count number of classes in dataset
DATASET_PATH = "train"
if os.path.exists(DATASET_PATH):
    classes = os.listdir(DATASET_PATH)
else:
    classes = ["Unknown"]

num_classes = len(classes)

# Load model
model = FaceRecognitionModel(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ✅ Step 4: Define Image Processing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ✅ Step 5: Real-Time Webcam Face Recognition
st.write("## 🎥 Live Face Recognition")
run = st.button("▶️ Start Live Face Recognition")

if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Unable to access webcam!")
    else:
        st.success("📷 Webcam Activated! Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

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

        cv2.imshow("Real-Time Face Recognition", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    st.warning("🔴 Webcam Closed!")
