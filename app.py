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

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('adr_data.db')
c = conn.cursor()

# Create tables if they don't exist
c.execute('''CREATE TABLE IF NOT EXISTS ADRInfo
             (id INTEGER PRIMARY KEY AUTOINCREMENT, medicine TEXT, adr TEXT)''')
c.execute('''CREATE TABLE IF NOT EXISTS MedicalReports
             (id INTEGER PRIMARY KEY AUTOINCREMENT, file_name TEXT, file_data BLOB)''')
conn.commit()

# Load your ADR data into the SQLite database (run this once to populate the table)
adr_data = pd.read_csv(r"C:\Users\Tanusree\Desktop\NEURODEGENERATION\New folder\frontendwork\drugs_side_effects_drugs_com.csv")
for index, row in adr_data.iterrows():
    c.execute("INSERT INTO ADRInfo (medicine, adr) VALUES (?, ?)", (row['Medicine'], row['ADR']))
conn.commit()

# Load your trained segmentation model
class YourSegmentationModel(nn.Module):
    def __init__(self):
        super(YourSegmentationModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Input: (3, 256, 256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by factor of 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Output: (32, 128, 128)
        self.fc1 = nn.Linear(32 * 64 * 64, 256)  # Adjusted input size for fc layer
        self.fc2 = nn.Linear(256, 10)  # Output layer for classification

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.pool(x)  # (16, 128, 128)
        
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)  # (32, 64, 64)

        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Initialize your model and load the weights
segmentation_model = YourSegmentationModel()

try:
    # Load the state dict with strict=False to allow for missing/unexpected keys
    loaded_weights = torch.load(r'C:\Users\Tanusree\Desktop\NEURODEGENERATION\New folder\frontendwork\segformer_finetuned.pth')
    missing_keys, unexpected_keys = segmentation_model.load_state_dict(loaded_weights, strict=False)

    # Check for missing or unexpected keys
    if missing_keys:
        print("Missing keys:", missing_keys)
    if unexpected_keys:
        print("Unexpected keys:", unexpected_keys)

except FileNotFoundError:
    st.error("The specified weights file was not found. Please check the path.")
except Exception as e:
    st.error(f"An error occurred while loading the model weights: {e}")

segmentation_model.eval()  # Set the model to evaluation mode

def identify_medicine(image):
    medicines = adr_data['Medicine'].dropna().astype(str).tolist()  # Ensure this is a list of strings
    
    # Process the inputs for CLIP
    inputs = processor(text=medicines, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    image_embeddings = outputs.image_embeds
    text_embeddings = outputs.text_embeds
    similarity = torch.cosine_similarity(image_embeddings, text_embeddings)
    best_match_index = similarity.argmax().item()
    identified_medicine = medicines[best_match_index]  # Ensure this is indexed correctly

    return identified_medicine

def preprocess_image_for_segmentation(image):
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to 256x256
        transforms.ToTensor(),
    ])
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

def segment_image(image):
    # Convert RGBA to RGB if necessary
    if image.mode == 'RGBA':
        image = image.convert('RGB')  # Convert to RGB

    image_tensor = preprocess_image_for_segmentation(image)
    with torch.no_grad():
        segmentation_output = segmentation_model(image_tensor)

    print(f"Segmentation output shape before postprocessing: {segmentation_output.shape}")
    
    segmented_image = postprocess_segmentation(segmentation_output)

    # Check if the segmented image is a 2D array
    if len(segmented_image.shape) == 2:  # [height, width]
        segmented_image_display = (segmented_image * 255).astype('uint8')
    elif len(segmented_image.shape) == 3 and segmented_image.shape[0] == 1:  # [1, height, width]
        segmented_image_display = (segmented_image[0] * 255).astype('uint8')  # Remove channel dimension
    else:
        raise ValueError("Segmented image is not in the expected format.")

    return Image.fromarray(segmented_image_display)

def postprocess_segmentation(segmentation_output):
    # Check for output dimensions
    if segmentation_output.dim() == 3:  # [batch_size, height, width]
        segmentation_mask = segmentation_output[0].cpu().numpy()  # Take the first item from batch
    elif segmentation_output.dim() == 4:  # [batch_size, num_classes, height, width]
        segmentation_mask = segmentation_output.argmax(dim=1).squeeze().cpu().numpy()  # Shape: [height, width]
    else:
        raise ValueError("Unexpected shape for segmentation mask.")
    
    return segmentation_mask


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

    # Insert the medical report into the SQLite database
    c.execute("INSERT INTO MedicalReports (file_name, file_data) VALUES (?, ?)", (report_file_name, report_file_data))
    conn.commit()

    st.success("Your medical report has been saved successfully.")





