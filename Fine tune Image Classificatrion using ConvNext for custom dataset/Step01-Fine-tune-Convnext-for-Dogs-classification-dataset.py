# Dataset : https://www.kaggle.com/datasets/muhammadhananasghar/9-dogs-breeds-identification-classification
import torch
from datasets import load_dataset 

# load the dataset and split into train and test sets
dataset = load_dataset("imagefolder", data_dir="D:/Data-Sets-Image-Classification/9 dogs Breeds")

print("Dataset : ")
print(dataset)

# Split the dataset into train and test sets
split_dataset = dataset["train"].train_test_split(test_size=0.2,  seed=42)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

# Print the sizes of the train and test sets
print("Train dataset size: ", len(train_dataset))
print("Test dataset size: ", len(test_dataset))

print("Train dataset : ")
print(train_dataset)
print("Test dataset : ")
print(test_dataset)

# Print the train key 
print("****************************************************")
print("Train dataset keys : ")
print(dataset["train"].features)

# Lets visual the first train example :

from PIL import Image

exmple = dataset["train"][0]
first_image = exmple["image"]
first_lebel = exmple["label"]

print(type(first_image))
print("Label value of the first image : ", str(first_lebel))

first_image.show()

# Get the class names
labels = dataset["train"].features["label"].names 
print("Labels - list of the class names : ")
print(labels)

# The labels are presened as intgers, but we can turm them into actual class names using the labels list
id2label = {k:v for k,v in enumerate(labels)}
label2id = {v:k for k,v in enumerate(labels)}
print("id2label : ")
print(id2label) 

# Display the first image  label :
print("Label of the first image : ", labels[first_lebel])

# -----------------------------------------------------------------------------------------------

# Process the dataset for the model :

from transformers import AutoImageProcessor
image_processor = AutoImageProcessor.from_pretrained("facebook/convnext-base-224")

print("Image processor : ")
print(image_processor)


# Image transformation : 

from torchvision.transforms import ( Compose , Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor)

# Noremalize the images by the image_mean and image_std
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

# RandomResizedCrop : is for 2 tasks : resize the images to 224X224 and data augmentation by random cropping during training

transform = Compose([
    RandomResizedCrop(image_processor.size["shortest_edge"]),
                      RandomHorizontalFlip(),
                      ToTensor(),
                      normalize  # use the normalize we define earlier
])


def data_transform(examples):
    examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["image"]]
    return examples


# Apply the transformation to the train and test datasets :

processed_train_dataset = train_dataset.with_transform(data_transform)
processed_test_dataset = test_dataset.with_transform(data_transform)

print("Processed train and test dataset : ")
print(processed_train_dataset[[0]]) # look at the first example -> all normalized and transformed


# Create a Pytorch loader for the train and test datasets :

from torch.utils.data import DataLoader

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}



# Create the loader

train_dataloader = DataLoader(dataset=processed_train_dataset, collate_fn=collate_fn, batch_size=8, shuffle=True)
val_dataloader = DataLoader(dataset=processed_test_dataset, collate_fn=collate_fn, batch_size=8, shuffle=False)


# get the first batch of the train dataloader
batch = next(iter(train_dataloader))
for k, v in batch.items():
    print(f"{k}: {v.shape}")



# Define the model 

from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained(
    "facebook/convnext-base-224",
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True, # take our custom dataset size and classes instead of the 1000 classes of imagenet
)


from tqdm import tqdm
import os 

save_dir = "d:/temp/models/convnext-dogs-classification/checkpoints"
os.makedirs(save_dir, exist_ok=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.train()

# Init variables to track the loss and accuracy
best_loss = float("inf")
epochs_without_improvement = 0
patience = 10
max_epochs = 100

# Training loop
for epoch in range(max_epochs):
    print(f"Epoch {epoch + 1} / {max_epochs}")
    train_loss = 0.0
    train_correct = 0
    train_total = 0


    # Training step
    model.train()

    for batch in tqdm(train_dataloader , desc = "Training"):
        batch = {k: v.to(device) for k, v in batch.items()} # move batch to GPU

        optimizer.zero_grad() # clear the gradients
        outputs = model(pixel_values=batch["pixel_values"], labels=batch["labels"]) # forward pass
        loss , logits = outputs.loss, outputs.logits # get the loss and logits
        loss.backward() # backward pass
        optimizer.step() # update the weights

        # Metrics
        train_loss += loss.item() 
        train_total += batch["labels"].shape[0]
        train_correct += (logits.argmax(-1) == batch["labels"]).sum().item() # count the correct predictions

    
    # Calculate training matrics
    train_accuracy = train_correct / train_total
    avg_train_loss = train_loss / len(train_dataloader)
    print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")



    # Validation step
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0


    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            batch = {k: v.to(device) for k, v in batch.items()} # move batch to GPU
            outputs = model(pixel_values=batch["pixel_values"], labels=batch["labels"]) # forward pass
            loss , logits = outputs.loss, outputs.logits # get the loss and logits

            val_loss += loss.item()
            val_total += batch["labels"].shape[0]
            val_correct += (logits.argmax(-1) == batch["labels"]).sum().item()


    # Calculate validation metrics
    val_accuracy = val_correct / val_total
    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Check for improvement in validation loss
    if avg_val_loss < best_loss :
        best_loss = avg_val_loss 
        epochs_without_improvement = 0

        # Save the best model :
        checkpoint_path = os.path.join(save_dir , "best_model.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"New best model saved with validation loss: {best_loss:.4f}")
    else:
        epochs_without_improvement += 1
        print(f"No improvement in validation loss for {epochs_without_improvement} epochs")


    # Early stopping
    if epochs_without_improvement >= patience:
        print(f"Early stopping after {patience} epochs without improvement.")
        break



    
 






























