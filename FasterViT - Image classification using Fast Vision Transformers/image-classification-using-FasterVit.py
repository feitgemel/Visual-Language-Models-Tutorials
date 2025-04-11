import torch
from torchvision import transforms
from PIL import Image
from fastervit import create_model

# Define fastervit-0 model with 224X224 input size and 1000 classes

# load the pre-trained model
model = create_model("faster_vit_0_224",
                     pretrained=True,
                     model_path="d:/temp/models/faster_vit_0.pth.tar")

# Set the model to evaluation mode
model.eval()


# Define the image processing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# load and preprocess the image
image_path = "Visual-Language-Models-Tutorials/FasterViT - Image classification using Fast Vision Transformers/Basketball.jpg"
img = Image.open(image_path)

# Preprocess the image
input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

# move the input to the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_batch = input_batch.to(device)

# Perform inference
with torch.no_grad():
    output = model(input_batch)

print("Output for 1000 classes: ")
print(output)



# Analyze the output
# assuimg the output is a classification score for 1000 classes
prob = torch.nn.functional.softmax(output[0], dim=0)

# Get the top 5 categories per the output
top5_prob , top5_catid = torch.topk(prob, 5)
for i in range(top5_prob.size(0)):
    print(f"Category: {top5_catid[i]}, Probability: {top5_prob[i].item()}")


# Display the class names :
# Download the class labels from this url : 
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

import json 
import requests
response = requests.get(url)
class_names = response.text.splitlines()

print(class_names)

# Create a dictionary to map class IDs to class names
class_idx = {i: class_names[i] for i in range(len(class_names))}

# save the dictionary to a json file
with open("d:/temp/imagenet_class_index.json", "w") as f:
    json.dump(class_idx, f)

# Load the class index from the json file
with open("d:/temp/imagenet_class_index.json", "r") as f:
    idx_to_labels = json.load(f)

print("idx_to_labels:")
print(idx_to_labels)

# Post process the results :
probs = torch.nn.functional.softmax(output[0], dim=0)

# get the top 5 categories per the output
top5_prob, top5_catid = torch.topk(probs, 5)
for i in range(top5_prob.size(0)):
    class_id = top5_catid[i].item()
    class_name = idx_to_labels[str(class_id)]
    probability = top5_prob[i].item()
    print(f"Class ID: {class_id}, Class Name: {class_name}, Probability: {probability:.4f}")



# Display the image with the highest predicted class label
from PIL import Image 
import matplotlib.pyplot as plt

plt.imshow(img)
plt.axis('off')

top_prob , top_catid = torch.topk(prob, 1)
predicted_class = top_catid[0].item()
predicted_class_name = idx_to_labels[str(predicted_class)]
probability = top_prob[0].item()

plt.text(20,20, f"Predicted: {predicted_class_name} ({probability:.4f})", color='white', backgroundcolor='black', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))

plt.show()

