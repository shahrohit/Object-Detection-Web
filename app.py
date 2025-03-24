import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load YOLO model
model = YOLO("../yolov10s.pt")

# ---- Set Custom Logo & Title ----
st.set_page_config(page_title="AI Object Detection", page_icon="logo.png", layout="wide")

# Show custom header image (replace 'header_image.jpg' with your own image)
st.image("header_image.png")

# Streamlit UI
st.write("### Upload an image to detect objects.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"],accept_multiple_files=False)

allowed_types = ["image/jpeg", "image/png", "image/jpg"]
if uploaded_file is not None:

    if uploaded_file.type not in allowed_types:
        st.error("Invalid file type! Please upload an image (JPG, PNG, JPG).")
    else:
        # Convert to OpenCV format
        image = Image.open(uploaded_file)
        image_np = np.array(image)  # Convert to NumPy array
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert to BGR for YOLO
        detected_image = image_np.copy()  # Copy for drawing detections

        # Perform object detection
        results = model(image_bgr)

        # Draw bounding boxes
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = box.conf[0].item()
                class_name = model.names[class_id]

                # Draw bounding box and label
                cv2.rectangle(detected_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(detected_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Display images side by side
        col1, col2 = st.columns(2)

        with col1:
            st.image(image_np, caption="Original Image", use_container_width=True)

        with col2:
            st.image(detected_image, caption="Detected Objects", use_container_width=True)
