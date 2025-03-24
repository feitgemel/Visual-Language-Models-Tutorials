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



    def predict_audio(self, audio_array):
        """Predict the label for a given audio array"""
        # Preprocess the audio
        required_length = self.sample_rate  # 1 second = 16,000 samples
        if len(audio_array) > required_length:
            audio_array = audio_array[:required_length]  # Trim to 1 second
        elif len(audio_array) < required_length:
            # Pad with zeros if shorter than 1 second
            padding = np.zeros(required_length - len(audio_array), dtype=audio_array.dtype)
            audio_array = np.concatenate([audio_array, padding])

        #print("Start predict")
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

        # Check if confidence is above the threshold
        if confidence >= 0.8:
            predicted_label = self.model.config.id2label[predicted_id]
            print(f"Predicted label: {predicted_label}, Confidence: {confidence:.2f} (Good prediction)")
            return predicted_label
        else:
            #print(f"Confidence: {confidence:.2f} (Prediction not good enough)")
            return None


    def audio_capture_callback(self, indata, frames, time, status):
        """Callback for audio input stream"""
        if status:
            print(f"Audio input stream error: {status}")
            return

        # Check if sound is above threshold
        amplitude = np.abs(indata).mean()
        
        # Debug print
        #print(f"Amplitude: {amplitude}")
        
        if amplitude > self.threshold:
            #print("Speech detected!")
            self.current_buffer.extend(indata.flatten())
            self.is_recording = True
        elif self.is_recording:
            # If recording and now silent, process the audio
            if len(self.current_buffer) > 0:
                # Convert to numpy array
                audio_data = np.array(self.current_buffer)
                
                # Save and predict
                self.process_utterance(audio_data)
                
                # Reset
                self.current_buffer = []
                self.is_recording = False

    def process_utterance(self, audio_data):
        """Process and save an utterance"""
        # Ensure 16kHz
        if len(audio_data) > 0:
            # Generate a unique filename
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"utterance_{timestamp}.wav"
            
            # Save the audio file
            sf.write(filename, audio_data, self.sample_rate)
            
            # Predict
            try:
                label = self.predict_audio(audio_data)
                #print(f"Predicted word: {label}")
            except Exception as e:
                print(f"Prediction error: {e}")

    def start_listening(self):
        """Start listening to microphone input"""
        print("Starting microphone listening...")
        print("Speak now. Words will be recorded and classified.")
        
        # List available input devices
        print("Available input devices:")
        print(sd.query_devices())
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate, 
                channels=1,  # Mono
                dtype='float32',
                callback=self.audio_capture_callback,
                device=None  # Let system choose default input device
            ):
                # Keep the stream open
                while True:
                    sd.sleep(1000)  # Sleep for 1 second
        except Exception as e:
            print(f"Error in audio stream: {e}")

def main():
    # Path to your trained model
    model_path = "d:/temp/models/wav2vec2-base-speech-commands/checkpoint-39780"
    
    # Create and start the audio classifier
    classifier = AudioClassifier(model_path)
    classifier.start_listening()

if __name__ == "__main__":
    main()