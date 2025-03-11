import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import streamlit as st

# Load trained model
MODEL_PATH = "models/face_recognition_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define CNN Model
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

# Load dataset classes
DATASET_PATH = "train"
if os.path.exists(DATASET_PATH):
    classes = os.listdir(DATASET_PATH)
else:
    classes = ["Unknown"]

# Load model
num_classes = len(classes)
model = FaceRecognitionModel(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Initialize webcam
cap = cv2.VideoCapture(0)

# Streamlit UI
st.title("🟢 AI Face Recognition - CareerUpskillers")
st.write("📞 Contact: [WhatsApp](https://wa.me/917975931377) | [Website](https://www.careerupskillers.com)")

# Start button
if st.button("Start Face Recognition"):
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Camera not detected!")
            break

        # Convert image
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            predicted_class = torch.argmax(output).item()

        label = classes[predicted_class] if predicted_class < len(classes) else "Unknown"

        # Display result
        cv2.putText(frame, f"Prediction: {label}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        st.image(frame, channels="BGR", use_column_width=True)

        if st.button("Stop"):
            break

cap.release()
cv2.destroyAllWindows()
st.write("🔴 Webcam stopped.")
