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

# Specify paths 
model_dir = "D:\Temp/5 vehichles/vit_custom"

path_to_data = "D:/Data-Sets-Image-Classification/5 vehichles for classification"
test_dir = os.path.join(path_to_data,"test")

# load the saved model and image processor
model = ViTForImageClassification.from_pretrained(model_dir)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

from transformers import ViTImageProcessor
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# Load the class names
test_dataset = ImageFolder(test_dir)
class_names = test_dataset.classes
#print(class_names)

# Load 6 random images from the test folder with their categoris :

all_images = [
    (os.path.join(root, file), os.path.basename(root))
    for root , _, files in os.walk(test_dir)
    for file in files if file.endswith(('.png', '.jpg', '.jpeg'))
]


random_image_paths = random.sample(all_images, 6)

#print(random_image_paths)

# Display the result :
fig , axes = plt.subplots(2, 3, figsize=(15,10)) # Create a grid for 6 images
axes = axes.flatten()

for idx , (image_path , true_label) in enumerate(random_image_paths):

    # load and preprocess the image
    sample_image = Image.open(image_path).convert("RGB")
    processed_sample = image_processor(sample_image, return_tensors="pt").to(device)

    # Predict the label
    with torch.no_grad(): # Disable gradient computation for inference 
        outputs = model(**processed_sample)

    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    predicted_label = class_names[predicted_class]

    #Display the image :
    axes[idx].imshow(sample_image)
    axes[idx].axis('off')
    axes[idx].set_title(f"True: {true_label}\nPred: {predicted_label}", fontsize=12)

# Adjust spaceing and display the plot 
plt.tight_layout() 
plt.show()

