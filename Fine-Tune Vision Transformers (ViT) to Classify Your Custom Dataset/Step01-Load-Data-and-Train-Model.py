# Dataset : https://www.kaggle.com/datasets/mrtontrnok/5-vehichles-for-multicategory-classification

import os 
import cv2
import numpy as np 
import random
import torch 
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor
from transformers import ViTFeatureExtractor, ViTForImageClassification, TrainingArguments, Trainer
import evaluate
from PIL import Image
import matplotlib.pyplot as plt

# load the VIT image processor
from transformers import ViTImageProcessor

# Early stopping
from transformers import EarlyStoppingCallback

# Specify paths :
path_to_data = "D:/Data-Sets-Image-Classification/5 vehichles for classification"
train_dir = os.path.join(path_to_data,"train")
val_dir = os.path.join(path_to_data,"validation")
test_dir = os.path.join(path_to_data,"test")

data_dir = "d:/temp/5 vehichles"

# load the VIT image process
# Hugging face : Google/vit-base-patch16-224-in21K
model_id = "google/vit-base-patch16-224-in21k"
image_processor = ViTImageProcessor.from_pretrained(model_id)

# Custom transformation pipeline for the dataset 

def transform(image):
    inputs = image_processor(image , return_tensors="pt")
    return inputs["pixel_values"].squeeze(0) # remove batch for Dataloader 

# load the datasets
train_dataset = ImageFolder(train_dir, transform=transform)
val_dataset = ImageFolder(val_dir, transform=transform)
test_dataset = ImageFolder(test_dir, transform=transform)

print(train_dataset)

# Collate function to handle batches in the Trainer 
def collate_fn(batch):
    images , labels = zip(*batch)
    return{
        "pixel_values": torch.stack(images),
        "labels": torch.tensor(labels)
    }

# load evaluation metric
metric = evaluate.load("accuracy")

def compute_metrics(p):
    predictions = np.argmax(p.predictions , axis=1)
    references = p.label_ids 
    return metric.compute(predictions=predictions, references=references)

# Prepare the model
num_classes = len(train_dataset.classes)

model = ViTForImageClassification.from_pretrained(
    model_id,
    num_labels=num_classes
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define train arguments :
# 200 epochs with early stopping 

training_args = TrainingArguments(
    output_dir= data_dir + "/vit_custom",
    per_device_train_batch_size=16,    
    per_device_eval_batch_size=16,
    eval_strategy="steps",
    num_train_epochs=200,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    load_best_model_at_end=True
)


# Define early stopping callback
early_stopping_callbak = EarlyStoppingCallback(
    early_stopping_patience=10, # Stop after no improvment for 10 evaluation steps,
    early_stopping_threshold=0.0 # improvment threshold (use 0.0 for exact match)
)

# Define the Trainer 

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=image_processor, # Use the VITImageProcessor
    callbacks=[early_stopping_callbak],
)

# Train the model 
train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train" , train_results.metrics)
trainer.save_metrics("train" , train_results.metrics)
trainer.save_state()

# Evaluate the model 
metrics = trainer.evaluate(test_dataset)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)













