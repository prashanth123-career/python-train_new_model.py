import os
import zipfile
import requests
import streamlit as st

# App Title
st.title("🔴 Live Face Recognition | CareerUpskillers")

# Contact Info
st.markdown("🚀 Developed by [CareerUpskillers](https://www.careerupskillers.com)")
st.markdown("📞 Contact: WhatsApp 917975931377")

# Dataset Direct Download Link (REPLACE WITH YOUR ACTUAL LINK)
DATASET_ZIP_URL = "https://drive.google.com/uc?id=1hjMmDWjWpJK4ewXWmEAr4i5Aq6newfIC"  # Replace this with actual Google Drive or GitHub raw file link
DATASET_PATH = "dataset"

# Function to download dataset
def download_dataset():
    st.warning("📂 Dataset not found. Downloading...")

    try:
        response = requests.get(DATASET_ZIP_URL, stream=True)
        response.raise_for_status()
        
        with open("dataset.zip", "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        # Extract the dataset
        with zipfile.ZipFile("dataset.zip", "r") as zip_ref:
            zip_ref.extractall(DATASET_PATH)

        st.success("✅ Dataset Downloaded & Extracted!")
    
    except Exception as e:
        st.error(f"❌ Failed to download dataset: {e}")
        st.stop()

# Check if dataset exists, otherwise download it
if not os.path.exists(DATASET_PATH):
    download_dataset()

# Now, you can proceed with the face recognition logic here
st.write("🎥 Ready for Face Recognition! (Continue with your main code)")
