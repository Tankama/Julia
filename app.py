import streamlit as st
from transformers import CLIPProcessor, CLIPModel
import torch
import pandas as pd
import sqlite3
from PIL import Image
from torchvision import transforms
import torch.nn as nn

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Connect to SQLite database (create if it doesn't exist)
conn = sqlite3.connect('adr_data.db')
c = conn.cursor()

# Create tables if they don't exist
c.execute('''CREATE TABLE IF NOT EXISTS ADRInfo
             (id INTEGER PRIMARY KEY AUTOINCREMENT, medicine TEXT, adr TEXT)''')
c.execute('''CREATE TABLE IF NOT EXISTS MedicalReports
             (id INTEGER PRIMARY KEY AUTOINCREMENT, file_name TEXT, file_data BLOB)''')
conn.commit()

# Load your ADR data into the SQLite database (upload CSV file)
uploaded_csv = st.file_uploader("Upload ADR CSV", type="csv")
if uploaded_csv is not None:
    adr_data = pd.read_csv(uploaded_csv)
    for index, row in adr_data.iterrows():
        c.execute("INSERT INTO ADRInfo (medicine, adr) VALUES (?, ?)", (row['Medicine'], row['ADR']))
    conn.commit()
    st.success("ADR data loaded into the database.")

# Load your trained segmentation model
class YourSegmentationModel(nn.Module):
    def __init__(self):
        super(YourSegmentationModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Initialize your model and load the weights
segmentation_model = YourSegmentationModel()
model_path = st.file_uploader("Upload your segmentation model weights", type=["pth"])
if model_path is not None:
    try:
        loaded_weights = torch.load(model_path)
        segmentation_model.load_state_dict(loaded_weights)
        st.success("Model weights loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model weights: {e}")

segmentation_model.eval()

# Function for identifying medicine
def identify_medicine(image):
    medicines = adr_data['Medicine'].dropna().astype(str).tolist()
    inputs = processor(text=medicines, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    image_embeddings = outputs.image_embeds
    text_embeddings = outputs.text_embeds
    similarity = torch.cosine_similarity(image_embeddings, text_embeddings)
    best_match_index = similarity.argmax().item()
    identified_medicine = medicines[best_match_index]
    return identified_medicine

# Image processing functions (remains unchanged)
# ... (insert your existing image processing functions here)

# Upload image for medicine identification
uploaded_medicine_file = st.file_uploader("Upload an image of the medicine", type=["jpg", "png", "jpeg"])
if uploaded_medicine_file is not None:
    medicine_image = Image.open(uploaded_medicine_file)
    st.image(medicine_image, caption="Uploaded Medicine Image", use_column_width=True)
    identified_medicine = identify_medicine(medicine_image)

    # Query ADR information from the database
    c.execute("SELECT adr FROM ADRInfo WHERE medicine=?", (identified_medicine,))
    adr_info = c.fetchone()

    st.write(f"Identified Medicine: {identified_medicine}")
    if adr_info:
        st.write(f"ADR Information: {adr_info[0]}")
    else:
        st.write("No ADR information found for this medicine.")

# Upload image for brain segmentation
uploaded_brain_file = st.file_uploader("Upload an image of the brain (MRI/CT)", type=["jpg", "png", "jpeg"], key="brain")
if uploaded_brain_file is not None:
    brain_image = Image.open(uploaded_brain_file)
    st.image(brain_image, caption="Uploaded Brain Image", use_column_width=True)
    segmented_image = segment_image(brain_image)
    st.image(segmented_image, caption="Segmented Brain Image", use_column_width=True)

# Upload medical report
report_file = st.file_uploader("Upload your medical report (Image/PDF)", type=["jpg", "png", "jpeg", "pdf"])
if report_file is not None:
    report_file_name = report_file.name
    report_file_data = report_file.read()
    c.execute("INSERT INTO MedicalReports (file_name, file_data) VALUES (?, ?)", (report_file_name, report_file_data))
    conn.commit()
    st.success("Your medical report has been saved successfully.")





