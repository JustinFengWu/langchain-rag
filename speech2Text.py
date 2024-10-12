# from openai import OpenAI
# import sounddevice as sd
# import numpy as np
# import scipy.io.wavfile as wav
# from pynput import keyboard


# transcript = ""

# # Set up OpenAI API key
# client = OpenAI()

# # Audio settings
# sample_rate = 16000  # Whisper works best with 16kHz
# recording = []
# is_recording = False

# def callback(indata, frames, time, status):
#     """This function will be called from the sounddevice during recording"""
#     if status:
#         print(status)
#     # Append the audio data to the recording array
#     recording.append(indata.copy())

# def start_recording():
#     global recording
#     recording.clear()  # Clear previous recordings
#     print("Recording started...")
#     # Use a non-blocking stream to record audio
#     stream = sd.InputStream(callback=callback, channels=1, samplerate=sample_rate, dtype=np.int16)
#     stream.start()
#     return stream

# def stop_recording(stream, filename="output.wav"):
#     print("Recording stopped.")
#     stream.stop()  # Stop the non-blocking stream
#     stream.close()

#     # Convert the list of numpy arrays to a single numpy array
#     recorded_data = np.concatenate(recording, axis=0)

#     # Save the recording to a .wav file
#     wav.write(filename, sample_rate, recorded_data)

# def transcribe_audio(file_path):
#     global transcript
#     print (file_path)
#     # print("Transcribing audio...")
#     # # Open the recorded file and send it to Whisper API for transcription
#     audio_file = open(file_path, "rb")
#     transcript = client.audio.transcriptions.create(
#         model="whisper-1",
#         file=audio_file,
#         response_format="text"
#         )
#     print("Transcription complete.")
#     print(transcript)
#     return transcript

# # Key press and release events to handle recording
# def on_press(key):
#     global is_recording, stream
#     try:
#         if key.char == '\\':  # Start/stop recording on backslash press
#             if not is_recording:
#                 is_recording = True
#                 stream = start_recording()
#             else:
#                 is_recording = False
#                 stop_recording(stream, "output.wav")
#                 # Transcribe the recorded audio
#                 transcription = transcribe_audio("output.wav")
#                 print("Transcription:", transcription)
#     except AttributeError:
#         pass

# def on_release(key):
#     if key == keyboard.Key.esc:
#         return False  # Stop listener on 'Esc' press

# def start():

#     print("Press '\\' to start/stop recording. Press 'Esc' to exit.")
#     # Start the keyboard listener for backslash key
#     with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
#         listener.join()

#     return transcript

from openai import OpenAI
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from pynput import keyboard

transcript = ""

# Set up OpenAI API key
client = OpenAI()

# Audio settings
sample_rate = 16000  # Whisper works best with 16kHz
recording = []
is_recording = False
should_terminate = False

def callback(indata, frames, time, status):
    """This function will be called from the sounddevice during recording"""
    if status:
        print(status)
    # Append the audio data to the recording array
    recording.append(indata.copy())

def start_recording():
    global recording
    recording.clear()  # Clear previous recordings
    print("Recording started...")
    # Use a non-blocking stream to record audio
    stream = sd.InputStream(callback=callback, channels=1, samplerate=sample_rate, dtype=np.int16)
    stream.start()
    return stream

def stop_recording(stream, filename="output.wav"):
    print("Recording stopped.")
    stream.stop()  # Stop the non-blocking stream
    stream.close()

    # Convert the list of numpy arrays to a single numpy array
    recorded_data = np.concatenate(recording, axis=0)

    # Save the recording to a .wav file
    wav.write(filename, sample_rate, recorded_data)

def transcribe_audio(file_path):
    global transcript
    print(file_path)
    # Open the recorded file and send it to Whisper API for transcription
    audio_file = open(file_path, "rb")
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="text"
    )
    print("Transcription complete.")
    print(transcript)
    return transcript

# Key press and release events to handle recording
def on_press(key):
    global is_recording, stream, should_terminate
    try:
        if key.char == '\\':  # Start/stop recording on backslash press
            if not is_recording:
                is_recording = True
                stream = start_recording()
            else:
                is_recording = False
                stop_recording(stream, "output.wav")
                # Transcribe the recorded audio
                transcription = transcribe_audio("output.wav")
                print("Transcription:", transcription)
                should_terminate = True  # Set termination flag to True
                return False  # Stop the listener to terminate the program
    except AttributeError:
        pass

def on_release(key):
    if key == keyboard.Key.esc:
        return False  # Stop listener on 'Esc' press

def start():
    global should_terminate
    print("Press '\\' to start/stop recording. Press 'Esc' to exit.")
    
    # Start the keyboard listener for backslash key
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()  # Wait for listener to stop

    # Exit when transcription is complete
    if should_terminate:
        return transcript

if __name__ == "__main__":
    start()