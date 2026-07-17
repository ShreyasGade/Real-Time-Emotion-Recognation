# streamlit_emotion_full.py

import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import time

# -----------------------------
# Load Models
# -----------------------------
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# Select 5 ma
# in emotions (you can also use all 7 if you want)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# -----------------------------
# Streamlit Layout
# -----------------------------
st.set_page_config(page_title="Real-Time Emotion Recognition", layout="centered")
st.title("Real-Time Emotion Recognition 🎭")
st.write("This app detects your emotions from your webcam feed.")

# Checkbox to start/stop webcam
run = st.checkbox("Start Webcam")

# Placeholder for video and probabilities
frame_placeholder = st.image([])
prob_placeholder = st.empty()

# -----------------------------
# Webcam Streaming
# -----------------------------
if run:
    camera = cv2.VideoCapture(0)
    frame_count = 0

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to capture video")
            break

        # Resize frame for faster processing
        frame = cv2.resize(frame, (400, 300))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        frame_clone = frame.copy()

        if len(faces) > 0:
            # Take the largest face
            faces = sorted(faces, reverse=True, key=lambda x: x[2]*x[3])[0]
            (fX, fY, fW, fH) = faces

            roi = gray[fY:fY+fH, fX:fX+fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Run prediction only every 3rd frame for speed
            if frame_count % 3 == 0:
                preds = emotion_classifier.predict(roi)[0]
            
            # Draw bounding box + main label
            label = EMOTIONS[np.argmax(preds)]
            cv2.rectangle(frame_clone, (fX, fY), (fX+fW, fY+fH), (0,0,255), 2)
            cv2.putText(frame_clone, label, (fX, fY-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # Create probability canvas (like OpenCV canvas)
            canvas = np.zeros((150, 400, 3), dtype="uint8")
            for i, (emotion, prob) in enumerate(zip(EMOTIONS, preds)):
                text = f"{emotion}: {prob*100:.1f}%"
                w = int(prob * 400)
                cv2.rectangle(canvas, (0, i*30), (w, i*30+25), (0,0,255), -1)
                cv2.putText(canvas, text, (5, i*30+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            # Convert canvas to RGB
            canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        else:
            preds = np.zeros(len(EMOTIONS))
            canvas_rgb = np.zeros((150, 400, 3), dtype=np.uint8)

        # Convert webcam frame to RGB
        frame_rgb = cv2.cvtColor(frame_clone, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb)
        prob_placeholder.image(canvas_rgb, caption="Emotion Probabilities")

        # Small delay for ~10 FPS
        time.sleep(0.1)
        frame_count += 1

        # Update checkbox state
        run = st.session_state.get("Start Webcam", True)

    camera.release()

# streamlit_emotion_full.py

# import streamlit as st
# import cv2
# import numpy as np
# from keras.models import load_model
# from keras.preprocessing.image import img_to_array
# import time

# # -----------------------------
# # Load Models
# # -----------------------------
# detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
# emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# face_detection = cv2.CascadeClassifier(detection_model_path)
# emotion_classifier = load_model(emotion_model_path, compile=False)

# # Select 5 main emotions (you can also use all 7 if you want)
# EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# # -----------------------------
# # Streamlit Layout
# # -----------------------------
# st.set_page_config(page_title="Real-Time Emotion Recognition", layout="centered")
# st.title("Real-Time Emotion Recognition 🎭")
# st.write("This app detects your emotions from your webcam feed.")

# # Checkbox to start/stop webcam
# run = st.checkbox("Start Webcam")

# # Placeholder for video and probabilities
# frame_placeholder = st.image([])
# prob_placeholder = st.empty()

# # -----------------------------
# # Webcam Streaming
# # -----------------------------
# if run:
#     camera = cv2.VideoCapture(0)
#     frame_count = 0

#     while run:
#         ret, frame = camera.read()
#         if not ret:
#             st.error("Failed to capture video")
#             break

#         # Resize frame for faster processing
#         frame = cv2.resize(frame, (400, 300))
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_detection.detectMultiScale(
#             gray,
#             scaleFactor=1.1,
#             minNeighbors=5,
#             minSize=(30, 30),
#             flags=cv2.CASCADE_SCALE_IMAGE
#         )

#         frame_clone = frame.copy()

#         if len(faces) > 0:
#             # Take the largest face
#             faces = sorted(faces, reverse=True, key=lambda x: x[2]*x[3])[0]
#             (fX, fY, fW, fH) = faces

#             roi = gray[fY:fY+fH, fX:fX+fW]
#             roi = cv2.resize(roi, (64, 64))
#             roi = roi.astype("float") / 255.0
#             roi = img_to_array(roi)
#             roi = np.expand_dims(roi, axis=0)

#             # Run prediction only every 3rd frame for speed
#             if frame_count % 3 == 0:
#                 preds = emotion_classifier.predict(roi)[0]
            
#             # Draw bounding box + main label
#             label = EMOTIONS[np.argmax(preds)]
#             cv2.rectangle(frame_clone, (fX, fY), (fX+fW, fY+fH), (0,0,255), 2)
#             cv2.putText(frame_clone, label, (fX, fY-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

#             # Create probability canvas (like OpenCV canvas)
#             canvas = np.zeros((150, 400, 3), dtype="uint8")
#             for i, (emotion, prob) in enumerate(zip(EMOTIONS, preds)):
#                 text = f"{emotion}: {prob*100:.1f}%"
#                 w = int(prob * 400)
#                 cv2.rectangle(canvas, (0, i*30), (w, i*30+25), (0,0,255), -1)
#                 cv2.putText(canvas, text, (5, i*30+20),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

#             # Convert canvas to RGB
#             canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
#         else:
#             preds = np.zeros(len(EMOTIONS))
#             canvas_rgb = np.zeros((150, 400, 3), dtype=np.uint8)

#         # Convert webcam frame to RGB
#         frame_rgb = cv2.cvtColor(frame_clone, cv2.COLOR_BGR2RGB)
#         frame_placeholder.image(frame_rgb)
#         prob_placeholder.image(canvas_rgb, caption="Emotion Probabilities")

#         # Small delay for ~10 FPS
#         time.sleep(0.1)
#         frame_count += 1

#         # Update checkbox state
#         run = st.session_state.get("Start Webcam", True)

#     camera.release()
    
#     face_detection = cv2.CascadeClassifier(detection_model_path)
# emotion_classifier = load_model(emotion_model_path, compile=False)

# # -----------------------------
# # Load Saved Performance Metrics
# # -----------------------------
# st.sidebar.header("📊 Model Performance Metrics")
# try:
#     with open("models/training_metrics.txt", "r") as f:
#         metrics_text = f.read()
#     st.sidebar.text(metrics_text)
# except FileNotFoundError:
#     st.sidebar.warning("⚠️ Metrics file not found. Please train the model first.")
