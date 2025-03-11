import streamlit as st
import av
import cv2
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ----------------- CareerUpskillers Branding -----------------
st.set_page_config(page_title="AI Face Recognition - CareerUpskillers", page_icon="🤖")
st.markdown(
    """
    <h1 style="text-align: center; color: #ff5733;">🤖 AI Face Recognition System</h1>
    <h3 style="text-align: center;">🚀 Developed by <a href='https://www.careerupskillers.com' target='_blank'>CareerUpskillers</a></h3>
    <p style="text-align: center;">📞 Contact: <a href="https://wa.me/917975931377" target="_blank">WhatsApp 917975931377</a></p>
    <hr style="border:1px solid #ff5733;">
    """,
    unsafe_allow_html=True
)

# ----------------- Model Download Setup -----------------
MODEL_URL = "https://drive.google.com/uc?id=1DF72bjGVnN6iNJrv6B18XulSdiCHBYtR"
MODEL_PATH = os.path.join("models", "face_recognition_model.pth")
os.makedirs("models", exist_ok=True)

if not os.path.exists(MODEL_PATH):
    st.info("🔄 Downloading model... Please wait.")
    try:
        import requests
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            st.success("✅ Model downloaded successfully!")
            st.info("Please refresh the page to load the model.")
            st.stop()
        else:
            st.error("❌ Failed to download model. Please check the URL.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Exception during download: {e}")
        st.stop()

# ----------------- Define Model Architecture -----------------
class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognitionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ----------------- Hard-Code Classes -----------------
# Here we know there are two classes, "known" and "unknown"
num_classes = 2
classes = ["known", "unknown"]

# ----------------- Load the Trained Model -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FaceRecognitionModel(num_classes=num_classes).to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
else:
    st.error(f"❌ Model file '{MODEL_PATH}' not found!")
    st.stop()

# ----------------- Define Image Transformations -----------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ----------------- Real-Time Video Processing using streamlit-webrtc -----------------
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.device = device
        self.model = model
        self.transform = transform
        self.classes = classes

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert the video frame to a numpy array
        img = frame.to_ndarray(format="bgr24")
        # Convert to PIL Image, then transform
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(img_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
        label = self.classes[predicted_class] if predicted_class < len(self.classes) else "Unknown"
        # Annotate the frame with the prediction
        cv2.putText(img, f"Prediction: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.header("📷 Real-Time Face Recognition (Webcam)")
st.write("Your webcam feed will display the predicted class in real time.")

webrtc_streamer(key="face_recognition", rtc_configuration=RTC_CONFIGURATION, video_processor_factory=VideoProcessor)

# ----------------- Footer Branding -----------------
st.markdown(
    """
    <hr style="border:1px solid #ff5733;">
    <h4 style="text-align: center;">🚀 AI Starter Kit by CareerUpskillers</h4>
    <p style="text-align: center;">For more AI tools, visit <a href='https://www.careerupskillers.com' target='_blank'>CareerUpskillers</a></p>
    """,
    unsafe_allow_html=True
)
