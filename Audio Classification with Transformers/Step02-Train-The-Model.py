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

# =================================================================

# Preprocessing the data

from transformers import AutoFeatureExtractor 

feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)

max_duration = 1 # Most of the audio files are 1 second. Lets set is a same value for all the files

# function for prerpcess all the samples

def preprocess_function(examples) :
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate = feature_extractor.sampling_rate,
        max_length = int(feature_extractor.sampling_rate * max_duration), # same length for every audio file for train
        truncation = True,
    )
    return inputs 


# The feature extracture will return a list of numpy arrays for each example :

first_file_elements = train[:5]
print("====================================================================")
print("First 5 elements :")
print(first_file_elements)

print("====================================================================")
print("Preprocess function for 5 elements :")
tmp = preprocess_function(train[:5])
print(tmp)


# Apply the funcion on all the dataset using the map function

encoded_train = train.map(preprocess_function, remove_columns=["audio", "file"], batched=True)

encoded_validation = validation.map(preprocess_function, remove_columns=["audio", "file"], batched=True)


# Train the mode :

from transformers import AutoModelForAudioClassification , Trainer, TrainingArguments 

num_labels = train.features["label"].num_classes
print(num_labels)
print(train.features["label"])

# Load the base model :

model = AutoModelForAudioClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    label2id=label2id,
    id2label= id2label,
)

# Trainig configuration 

model_name = model_checkpoint.split("/")[-1]
print("Model name :" + model_name)
model_name = f"{model_name}-speech-commands"
print("Model name :" + model_name)

import os 
model_save_path = "d:/temp/models/" + model_name

# Create a folder 
os.makedirs(model_save_path, exist_ok=True)
print("model_save_path")
print(model_save_path)


# Set train paramerts :
args = TrainingArguments(
    output_dir=model_save_path,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=60, 
    warmup_ratio=0.1 ,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)



# Set evaluation for the end of each epoch

import numpy as np 

def compute_matrix(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

# Pass it to the Trainer

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_train,
    eval_dataset=encoded_validation,
    tokenizer = feature_extractor,
    compute_metrics=compute_matrix
)


trainer.train()



