import cv2
import numpy as np
import streamlit as st
from PIL import Image
import mediapipe as mp

st.title('Face Detection Application')

uploaded_image = st.file_uploader('Choose an Image:', type=['png', 'jpeg', 'jpg'])

options = st.selectbox('Choose Face Detection Method', ('OpenCV', 'MediaPipe'))

# Using OpenCV for face detection
def detect_face_opencv(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
    
    return img

# Using MediaPipe for face detection
def detect_face_mediapipe(img):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for MediaPipe

    with mp_face_detection.FaceDetection() as face_detection:
        results = face_detection.process(img_rgb)

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(img_rgb, detection)

    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)  # Convert back to BGR for display

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image = np.array(image)

    # Convert RGBA to RGB (if image has an alpha channel)
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR

    if options == 'OpenCV':
        result = detect_face_opencv(image)
    elif options == 'MediaPipe':
        result = detect_face_mediapipe(image)
    else:
        st.write('Try another one')

    st.image(result, channels='BGR', use_container_width=True)
