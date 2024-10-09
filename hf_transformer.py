import cv2
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import time  # For managing the time interval
from collections import defaultdict  # For emotion counting

def run_emotion_detection():
    # Load the pre-trained CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Capture webcam video
    cap = cv2.VideoCapture(0)

    # Define emotion labels corresponding to the indices
    emotion_labels = ["happy", "sad", "neutral", "annoyed"]

    # Dictionary to store emotion counts
    emotion_counter = defaultdict(int)

    # Start time for emotion recording
    last_write_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame to PIL Image for processing
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Prepare inputs for the model
        inputs = processor(text=emotion_labels, images=image, return_tensors="pt", padding=True)

        # Run the model prediction
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the probabilities
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        # Get the index of the highest probability
        emotion_index = torch.argmax(probs, dim=1).item()
        emotion_text = emotion_labels[emotion_index]

        # Count the detected emotion
        emotion_counter[emotion_text] += 1

        # Show the frame with detected emotion
        cv2.putText(frame, emotion_text.capitalize(), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)

        # Write emotion text to file every 5 seconds
        current_time = time.time()
        if current_time - last_write_time >= 5:
            with open("emotion_log.txt", "w") as f:
                f.write(emotion_text)
            last_write_time = current_time  # Reset the timer

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close video capture after exiting loop
    cap.release()
    cv2.destroyAllWindows()
