import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import librosa
import queue
import threading
import time
import torch.nn.functional as F
from pynput.keyboard import Controller, Key
import pygetwindow as gw
import pyautogui

class AudioClassifier:
    def __init__(self, model_path):
        # Load model and feature extractor
        print("Initializing model...")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
        self.model = AutoModelForAudioClassification.from_pretrained(model_path)
        self.model.eval()

        # Audio recording parameters
        self.sample_rate = 16000  # 16kHz
        self.block_duration = 1.0  # 1 second blocks
        self.threshold = 0.05  # Lowered threshold for more sensitivity
        self.silence_duration = 1.0  # Silence duration to consider end of utterance

        # Queues and flags
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.current_buffer = []

        # Keyboard controller
        self.keyboard = Controller()

    def focus_on_window(self, window_title):
        """Focus on a window with the given title."""
        windows = gw.getWindowsWithTitle(window_title)
        if windows:
            window = windows[0]
            print(f"Focusing on window: {window.title}")
            window.activate()
            pyautogui.click(window.left + 10, window.top + 10)  # Ensure the window is focused
        else:
            print(f"Window with title '{window_title}' not found.")

    def simulate_key_press(self, key):
        """Simulate a short key press."""
        self.keyboard.press(key)
        time.sleep(0.1)  # Hold the key for a short time
        self.keyboard.release(key)

    def predict_audio(self, audio_array):
        """Predict the label for a given audio array."""
        # Preprocess the audio
        required_length = self.sample_rate  # 1 second = 16,000 samples
        if len(audio_array) > required_length:
            audio_array = audio_array[:required_length]  # Trim to 1 second
        elif len(audio_array) < required_length:
            # Pad with zeros if shorter than 1 second
            padding = np.zeros(required_length - len(audio_array), dtype=audio_array.dtype)
            audio_array = np.concatenate([audio_array, padding])

        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=self.sample_rate,
            max_length=int(self.sample_rate * 1),
            truncation=True,
            return_tensors="pt"
        )

        # Perform inference
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probabilities = F.softmax(logits, dim=-1)  # Convert logits to probabilities
            predicted_id = torch.argmax(probabilities, axis=-1).item()
            confidence = probabilities[0, predicted_id].item()  # Extract confidence for the predicted label

        if confidence >= 0.8:
            predicted_label = self.model.config.id2label[predicted_id]
            print(f"Predicted label: {predicted_label}, Confidence: {confidence:.2f} (Good prediction)")
            return predicted_label
        else:
            return None

    def audio_capture_callback(self, indata, frames, time, status):
        """Callback for audio input stream."""
        if status:
            print(f"Audio input stream error: {status}")
            return

        amplitude = np.abs(indata).mean()
        if amplitude > self.threshold:
            self.current_buffer.extend(indata.flatten())
            self.is_recording = True
        elif self.is_recording:
            if len(self.current_buffer) > 0:
                audio_data = np.array(self.current_buffer)
                self.process_utterance(audio_data)
                self.current_buffer = []
                self.is_recording = False

    def process_utterance(self, audio_data):
        """Process and save an utterance."""
        if len(audio_data) > 0:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"utterance_{timestamp}.wav"
            sf.write(filename, audio_data, self.sample_rate)

            try:
                label = self.predict_audio(audio_data)
                if label == "left":
                    self.simulate_key_press(Key.left)
                elif label == "right":
                    self.simulate_key_press(Key.right)
                elif label == "up":
                    self.simulate_key_press(Key.up)
                elif label == "down":
                    self.simulate_key_press(Key.down)
            except Exception as e:
                print(f"Prediction error: {e}")

    def start_listening(self):
        """Start listening to microphone input."""
        print("Starting microphone listening...")
        print("Speak now. Words will be recorded and classified.")

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                callback=self.audio_capture_callback,
                device=None
            ):
                while True:
                    sd.sleep(1000)
        except Exception as e:
            print(f"Error in audio stream: {e}")


def main():
    model_path = "d:/temp/models/wav2vec2-base-speech-commands/checkpoint-39780"
    classifier = AudioClassifier(model_path)

    # Focus on the desired window
    WindowName = 'Stella 6.5.2: "Pac-Man (1982) (Atari)'
    #WindowName = 'test.txt - Notepad'
    
    classifier.focus_on_window(WindowName)

    # Start listening to audio
    classifier.start_listening()

if __name__ == "__main__":
    main()
