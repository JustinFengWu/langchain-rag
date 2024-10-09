import threading
from hf_transformer import run_emotion_detection
from query_data import run_chatbot  # Assuming both scripts contain the main functions

def emotion_thread():
    # This will run emotion detection, which writes the emotion to a file
    run_emotion_detection()

def chatbot_thread():
    # Read emotion from file and start chatbot
    run_chatbot()

if __name__ == "__main__":
    # Start emotion detection thread
    emotion_detection_thread = threading.Thread(target=emotion_thread)
    emotion_detection_thread.start()

    # Start chatbot thread
    chatbot_thread = threading.Thread(target=chatbot_thread)
    chatbot_thread.start()

    # Wait for both threads to finish
    emotion_detection_thread.join()
    chatbot_thread.join()

    print("Both processes have completed.")
