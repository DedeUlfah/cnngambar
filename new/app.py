import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model(r'C:\Users\Sinta\Documents\Tugas Akhir\Backup Plan\project-TA\models\sign_language_cnn_model.h5')

# Function to capture video from camera and make predictions
def capture_video():
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame for your model
        processed_frame = preprocess_frame(frame)

        # Predict using the model
        prediction = model.predict(np.expand_dims(processed_frame, axis=0))

        # Translate prediction to sign language alphabet
        alphabet = translate_prediction(prediction)

        # Display the frame and prediction
        frame = cv2.putText(frame, alphabet, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Convert the frame to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Use stframe to update the image in the app
        stframe.image(frame_rgb, channels="RGB")

    cap.release()

def preprocess_frame(frame):
    # Resize frame to the size your model expects
    frame = cv2.resize(frame, (64, 64))
    # Normalize the frame
    frame = frame / 255.0
    return frame

def translate_prediction(prediction):
    # Translate model prediction to human-readable alphabet
    # Assuming your model output is a probability distribution over 26 classes (A-Z)
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    predicted_index = np.argmax(prediction)
    return alphabet[predicted_index]

st.title("Sign Language Detection")
st.write("Click the button below to start capturing video and detecting sign language")

if st.button("Start Camera"):
    capture_video()
