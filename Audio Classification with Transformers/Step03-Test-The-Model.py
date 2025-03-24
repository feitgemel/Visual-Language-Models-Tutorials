import librosa.display
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf

#filename = "Visual-Language-Models-Tutorials/Audio Classification with Transformers/happy16k.wav"
#filename = "Visual-Language-Models-Tutorials/Audio Classification with Transformers/stop16k.wav"
filename = "Visual-Language-Models-Tutorials/Audio Classification with Transformers/up16k.wav"

# Load the audio file
audio , sr = librosa.load(filename)

# Play the audio file
print("Play the audio file")
sd.play(audio, samplerate=sr)

# Wait until the audio file is played
status = sd.wait()

# Display the audio file
plt.figure(figsize=(14, 5))
librosa.display.waveshow(audio, sr=sr)
plt.show()

# Run the prediction

# Choose a checkpoint :
local_model_path = "D:/Temp/Models/wav2vec2-base-speech-commands/checkpoint-39780"

import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

print("Load the model")
feature_extractor = AutoFeatureExtractor.from_pretrained(local_model_path)
model = AutoModelForAudioClassification.from_pretrained(local_model_path)

model.eval() # Set the model to evaluation mode

# load the labels 

label2id = model.config.label2id
print(label2id)
id2label = model.config.id2label
print(id2label)

# Preprocess the audio file
print(f"Load audio file : {filename}")
audio_array , sampling_rate = sf.read(filename)

if sampling_rate != feature_extractor.sampling_rate :
    raise ValueError(
        f"Sampling rate of the audio file is {sampling_rate} Hz does not match the model's expected sampling rate " 
        f"({feature_extractor.sampling_rate} Hz). Please resample the audio file."
    )

# Preprocess the audio file for the model
print("Preprocess the audio file for the model")
inputs = feature_extractor(
    audio_array,
    sampling_rate=feature_extractor.sampling_rate,
    max_length=feature_extractor.sampling_rate * 1, # Adjust to your model maximum duration - 1 second
    truncation=True,
    return_tensors="pt",
)

# Perform inference
print("Perform inference")
with torch.no_grad():
    logits = model(**inputs).logits
    predicted_class_idx = torch.argmax(logits , axis=-1).item()
    print("Predicted class index :")
    print(predicted_class_idx)


# Print the predicted class name
predicted_label = model.config.id2label[predicted_class_idx]
print(f"Predicted label : {predicted_label}")







