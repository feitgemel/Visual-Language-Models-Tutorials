import torch 
from datasets import load_dataset
from torchvision.transforms import Compose, Normalize , ToTensor, Resize
import matplotlib.pyplot as plt
from PIL import Image

dataset = load_dataset("imagefolder", data_dir="D:/Data-Sets-Image-Classification/9 dogs Breeds")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Get the class names
labels = dataset["train"].features["label"].names 
print("Labels - list of the class names : ")
print(labels)

# The labels are presened as intgers, but we can turm them into actual class names using the labels list
id2label = {k:v for k,v in enumerate(labels)}
label2id = {v:k for k,v in enumerate(labels)}
print("id2label : ")
print(id2label) 


from transformers import AutoImageProcessor
image_processor = AutoImageProcessor.from_pretrained("facebook/convnext-base-224")


transform = Compose([
    Resize(image_processor.size["shortest_edge"]), 
    ToTensor(),
    Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
])

# Define the model 

from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained(
    "facebook/convnext-base-224",
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True, # take our custom dataset size and classes instead of the 1000 classes of imagenet
)

#load the saved model 
checkpoint_path = "D:/Temp/Models/convnext-dogs-classification/checkpoints/best_model.pth"

# load the state dict
state_dict = torch.load(checkpoint_path) 

# Load the weights into the model
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

model.to(device)

# load and preprocess the image
image_path = "Visual-Language-Models-Tutorials/Fine tune Image Classificatrion using ConvNext for custom dataset/Dori.jpg"

image = plt.imread(image_path)
input_image = transform(Image.fromarray(image).convert("RGB")).unsqueeze(0).to(device) # add batch dimension and move to device


# make prediction
with torch.no_grad():
    outputs = model(pixel_values=input_image)
    logits = outputs.logits
    predicted_class_id = logits.argmax(-1).item()
    print(f"Predicted class id: {predicted_class_id}")  
    predicted_label = id2label[predicted_class_id]


# Display the image and the predicted label
plt.imshow(image)
plt.title(f"Predicted label: {predicted_label}")
plt.axis("off")
plt.show()



























