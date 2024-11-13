import streamlit as st
from fastai.vision.all import load_learner, PILImage
from ultralytics import YOLO
from PIL import Image, ImageDraw
import tempfile

# App title and description
st.title("Car Color Detection and People Counter")
st.write("Upload an image to detect the car color and count the number of people present.")

# Load models
@st.cache_resource
def load_models():
    car_color_model = load_learner('car_color_model.pkl')
    person_detection_model = YOLO('yolov8n.pt')  # Ensure yolov8n.pt is accessible
    return car_color_model, person_detection_model

car_color_model, person_detection_model = load_models()

# File upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preview the uploaded image
    st.image(uploaded_file, caption="Uploaded Image Preview", use_column_width=True)
    
    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        img_path = temp_file.name

    # Load image for FastAI model
    img = PILImage.create(img_path)
    
    # Car color prediction
    car_pred, car_pred_idx, car_probs = car_color_model.predict(img)

    # Draw bounding box for car color
    img_with_boxes = img.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    car_box_color = "red" if car_pred == "blue" else "blue"
    draw.rectangle([(10, 10), (img.width - 10, img.height - 10)], outline=car_box_color, width=5)
    
    # People detection
    results = person_detection_model(img_path)
    people_count = sum([1 for result in results[0].boxes if result.cls == 0])  # Class 0 is "person" in YOLO's COCO dataset
    
    # Display results
    st.image(img_with_boxes, caption="Image with Car Color Prediction", use_column_width=True)
    st.write(f"**Predicted Car Color:** {car_pred}")
    st.write(f"**Probability:** {car_probs[car_pred_idx]:.4f}")
    st.write(f"**People Detected:** {people_count}")
