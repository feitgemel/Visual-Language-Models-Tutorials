import torch 
from torchvision import transforms as T  
import timm 
from PIL import Image

# Get list of models :
print(timm.list_models())

model_name = "convnext_base_384_in22ft1k"

# Create a model instance
model = timm.create_model(model_name, pretrained=True)

model.eval() # set to inference mode

trans_ = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD)
])

image = Image.open('Visual-Language-Models-Tutorials/Image Classification using ConvNeXt/Fintch.jpg')
#image = Image.open('Visual-Language-Models-Tutorials/Image Classificatrion using ConvNeXt/basketball.jpg')

transformed = trans_(image)
batch_of_img = transformed.unsqueeze(0)

with torch.no_grad():
    out = model(batch_of_img)

pred = out.argmax(dim=1)
print("Predicion value: ")
print(pred)

# what is class id 12 ?

# Imagenet labels
URL = "https://raw.githubusercontent.com/SharanSMenon/swin-transformer-hub/main/imagenet_labels.json" 

import json 
from urllib.request import urlopen

res = urlopen(URL)
classes = json.loads(res.read())
print(len(classes))

predicted_label = classes[pred.item()]
print(predicted_label)

# Display the image woth predicted label
import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))
plt.imshow(image)
plt.axis('off')
plt.title(f"Predicted label: {predicted_label}" , fontsize=16)
plt.show()





