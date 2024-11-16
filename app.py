import streamlit as st
from fastai.vision.all import load_learner, PILImage
from ultralytics import YOLO
from PIL import Image, ImageDraw
import tempfile

# App title and description
st.title("Car Color Detection and People Counter")
st.write("Upload an image to detect car color and count the number of people present.")

# Load models
@st.cache_resource
def load_models():
    car_color_model = load_learner('car_color_model.pkl')
    person_detection_model = YOLO('yolov8n.pt')  # Ensure yolov8n.pt is accessible
    return car_color_model, person_detection_model

car_color_model, person_detection_model = load_models()

# File upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.read())
        img_path = temp_file.name

    # Preview the uploaded image
    st.image(img_path, caption="Uploaded Image Preview", use_column_width=True)

    # Load image for FastAI model
    img = PILImage.create(img_path)

    # Car color prediction
    car_pred, car_pred_idx, car_probs = car_color_model.predict(img)
    car_box_color = "red" if car_pred == "blue" else "blue"

    # People detection using YOLOv8
    results = person_detection_model(img_path)

    # Extract bounding boxes and class information
    people_count = 0
    img_with_boxes = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img_with_boxes)

    for box in results[0].boxes.data:  # Iterate over detections
        x1, y1, x2, y2, confidence, cls = box.tolist()
        cls = int(cls)  # Convert to integer class
        if cls == 0:  # Class 0 corresponds to "person"
            people_count += 1
            draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)

    # Draw a bounding box for the car color prediction
    draw.rectangle([10, 10, img_with_boxes.width - 10, img_with_boxes.height - 10], outline=car_box_color, width=5)

    # Display results
    st.image(img_with_boxes, caption="Processed Image", use_column_width=True)
    st.write(f"**Predicted Car Color:** {car_pred}")
    st.write(f"**Probability:** {car_probs[car_pred_idx]:.4f}")
    st.write(f"**People Detected:** {people_count}")
