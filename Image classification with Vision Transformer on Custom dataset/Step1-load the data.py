# Dataset : https://www.kaggle.com/datasets/asadullahgalib/guava-disease-dataset

import torch
import matplotlib.pyplot as plt
from torch import nn 
from torchvision import transforms
import os 
import numpy as np 
from torchvision import datasets
from torch.utils.data import DataLoader
from torchinfo import summary

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

print("Torch version:", torch.__version__)

train_dir = 'D:/Data-Sets-Image-Classification/Guava Fruit Disease Dataset/train'
valid_dir = 'D:/Data-Sets-Image-Classification/Guava Fruit Disease Dataset/val'

NUM_WORKERS = 0

def create_dataloaders (
        train_dir: str,
        valid_dir: str,
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int = NUM_WORKERS
):
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    valid_data = datasets.ImageFolder(valid_dir, transform=transform)
    class_names = train_data.classes

    train_data_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)
    
    valid_data_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)
    
    return train_data_loader, valid_data_loader, class_names

IMG_SIZE = 224
BATCH_SIZE = 32
manual_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])
print(f"Manually created transform: {manual_transform}")





# Run the code :
train_data_loader, valid_data_loader, class_names = create_dataloaders(
    train_dir=train_dir,
    valid_dir=valid_dir,
    transform=manual_transform,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)

print(train_data_loader, valid_data_loader, class_names)


# Calculate number of images and batches
num_train_images = len(train_data_loader.dataset)
num_valid_images = len(valid_data_loader.dataset)
print(f"Number of training images: {num_train_images}")
print("Number of training batches:", len(train_data_loader))
print("=====================================================================")
print(f"Number of validation images: {num_valid_images}")
print("Number of validation batches:", len(valid_data_loader))


# Display one image for the train dataset :
# Get the first batch of images 
image_batch , label_batch = next(iter(train_data_loader))
image , label = image_batch[0], label_batch[0]
print("=======================================================================")
print(f"Image shape: {image.shape}")

# Convert the Pytorch tensor to numpy array
image_np = image.numpy()

# Reshape the image from (C, H, W) to (H, W, C) format
image_rearraged = np.transpose(image_np , (1 , 2, 0))

# Display the image
plt.title(class_names[label])
plt.imshow(image_rearraged)
plt.axis('off')
plt.show()


    
