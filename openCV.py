# import cv2
# from keras.models import load_model

# # Load a pre-trained emotion detection model (you'll need to find a model or train one)
# model = load_model('emotion_model.h5')

# # Capture webcam video (or use an image)
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()

#     # Preprocess frame and run it through the model
#     emotion = model.predict(frame)

#     # Show the frame with detected emotion
#     cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     cv2.imshow('frame', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
from keras.models import load_model

# Load a pre-trained emotion detection model
model = load_model('EmotionDetectionModel.h5')

# Capture webcam video
cap = cv2.VideoCapture(0)

# Define emotion labels corresponding to the indices
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess frame: convert to grayscale, resize, and normalize
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    resized_frame = cv2.resize(gray_frame, (48, 48))  # Resize to the input size of the model (e.g., 48x48)
    normalized_frame = resized_frame / 255.0  # Normalize pixel values to [0, 1]
    reshaped_frame = np.reshape(normalized_frame, (1, 48, 48, 1))  # Reshape for the model (1, height, width, channels)

    # Run the model prediction
    emotion_prediction = model.predict(reshaped_frame)
    emotion_index = np.argmax(emotion_prediction)  # Get the index of the highest probability
    emotion_text = emotion_labels[emotion_index]  # Map index to corresponding emotion label

    # Show the frame with detected emotion
    cv2.putText(frame, emotion_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
