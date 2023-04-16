import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import streamlit_webrtc
import av

st.title("Emotion Detector")

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('model.h5')

emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad']

webrtc_streamer = streamlit_webrtc.component.webrtc_streamer()

emotion_map = {
    0: 0,  # Angry
    1: 0,  # Angry (ค่าใหม่ที่จะแทนที่ 'Disgust' เป็น 'Angry')
    2: 3,  # Sad (ค่าใหม่ที่จะแทนที่ 'Fear' เป็น 'Sad')
    3: 1,  # Happy
    4: 2,  # Neutral
    5: 3,  # Sad
    6: 1,  # Happy (ค่าใหม่ที่จะแทนที่ 'Surprise' เป็น 'Happy')
}

def emotion_detector(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        img_gray = gray[y:y + h, x:x + w]
        img_gray = cv2.resize(img_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([img_gray]) != 0:
            img = img_gray.astype('float') / 255.0
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)

            prediction = classifier.predict(img)[0]
            mapped_prediction = emotion_map[prediction.argmax()]
            label = emotion_labels[mapped_prediction]

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(image, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return image

class VideoProcessor(streamlit_webrtc.VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = emotion_detector(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def app():
    webrtc_ctx = streamlit_webrtc.WebRtcMode.SENDRECV

    streamlit_webrtc.webrtc_streamer(
        key="emotion_detector",
        mode=webrtc_ctx,
        video_processor_factory=VideoProcessor,
    )

app()

