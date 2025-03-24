# Dataset : https://huggingface.co/datasets/google/speech_commands
# Model : https://huggingface.co/facebook/wav2vec2-base

from datasets import load_dataset
import evaluate
import aiohttp 

model_checkpoint = "facebook/wav2vec2-base"
batch_size = 32

train, validation = load_dataset("speech_commands" , "v0.02", split=["train" , "validation"],
                                 trust_remote_code=True,
                                 storage_options={'client_kwargs': {'timeout':aiohttp.ClientTimeout(total=3600)}})

print(train)

print(train.features["label"])
labels = train.features["label"].names

print(labels)

metric = evaluate.load("accuracy")


# Lets check a sample audio :

print("============================================================")
label2id = {label: labels.index(label) for label in labels}
id2label = {str(id): label for label , id in label2id.items()}

print(label2id)
print(id2label)

# Hear the sound :
import sounddevice as sd 
sample_audio = train[0]["audio"]

# Play the audio using sounddevice 
sd.play(sample_audio["array"], samplerate=sample_audio["sampling_rate"])
sd.wait()


import random

for _ in range(5):
    rand_idx = random.randint(0, len(train) -1 )
    example = train[rand_idx]
    audio = example["audio"]

    # Play the audio
    sd.play(audio["array"], samplerate=audio["sampling_rate"])
    sd.wait()

    print(f'Label: {id2label[str(example["label"])]}')
    print(f'Shape: {audio["array"].shape}, sampling rate: {audio["sampling_rate"]}')
    print()


