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


# move the input to the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

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

import cv2 
video_path = "Visual-Language-Models-Tutorials/FasterViT - Image classification using Fast Vision Transformers/Airplane.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error : Could not open video ....")
    exit()


# Define output screen size 
output_width = 640 
output_height = 480



# Preprocess each frame in the video :
while cap.isOpened():
    ret , frame = cap.read()
    if not ret:
        break 

    # Resize frame to desired output size
    frame = cv2.resize(frame , (output_width, output_height))

    # Convert frame to PIL image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Apply preprocessing 

    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)

    # Move the input to the GPU / CPU device 
    input_batch = input_batch.to(device)

    # Run the model 
    with torch.no_grad():
        output = model(input_batch)

    # Get predicted class 
    probs = torch.nn.functional.softmax(output[0], dim=0)

    top_prob , top_catid = torch.topk(probs, 1)
    predicted_class = top_catid[0].item()
    predicted_class_name = idx_to_labels[str(predicted_class)]
    probability = top_prob[0].item()

    # Display the classification result on the frame 
    cv2.putText(frame, f'Class: {predicted_class_name}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f'Probability: {probability}', (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # Display the frame 
    cv2.imshow("Video Classification ", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()




















































