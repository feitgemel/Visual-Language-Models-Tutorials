import torch
from torchvision import transforms
from fastervit import create_model
import os 
import cv2
import numpy as np

num_classes = 50

# Create model instance
model = create_model('faster_vit_0_224', pretrained=False)

# modify the last layer to match the number of classes in your dataset
model.head = torch.nn.Linear(model.head.in_features, num_classes)

# move the model into Gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# load the saved weights
model_path = 'd:/temp/models/star_wars_faster_vit_model.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval() # set the model to evaluation mode


# Define data transforms for the input image 
preprocess = transforms.Compose([
    transforms.ToPILImage(), # Convert the numpy array to PIL Image
    transforms.Resize((256)),
    transforms.CenterCrop((224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# function to load and preprocess the image
def load_image(image_path): 
    image = cv2.imread(image_path) # load the image using OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert BGR to RGB
    image = preprocess(image) # apply the transformations
    image = image.unsqueeze(0)
    image = image.to(device) # move the image to GPU if available

    return image


# function to predict the class of the image
def predict(image_path , model , class_names):
    image = load_image(image_path)
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        preducted_class = class_names[preds.item()]
    return preducted_class
    

from glob import glob

# path for test images - to get classes names from the folder names
testPath = "D:/Data-Sets-Image-Classification/Star-Wars-Characters-For-Classification/Test"

# Get the subfolder names (class names) from the test folder
class_names = [f for f in os.listdir(testPath) if os.path.isdir(os.path.join(testPath, f))]
print(class_names)

# define the number of classes
num_classes = len(class_names)

imagePath = "Visual-Language-Models-Tutorials/FasterViT - StarWars - Image classification on your Custom Dataset using Fast Vision Transformers/Yoda-Test-Image.jpg"
predicted_class = predict(imagePath, model, class_names)
print(f"Predicted class : {predicted_class}")


# Function to make prediction and draw the label on the image   

def predict_and_draw(image_path, model, class_names):
    # load the image 
    image = cv2.imread(image_path) # load the image using OpenCV
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert BGR to RGB
    input_tensor = preprocess(image_rgb) # apply the transformations
    input_tensor = input_tensor.unsqueeze(0) # add batch dimension
    input_tensor = input_tensor.to(device) # move the image to GPU if available

    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)
        predicted_class = class_names[preds.item()]

    # draw the label on the image
    text = f"Predicted: {predicted_class}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 3
    text_x , text_y = 10 , 50 

    cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 100, 100), font_thickness)

    # Display the image with the label
    cv2.imshow("Predicted Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the image with the label
    ouput_image_path = "D:/temp/predicted_image.jpg"
    cv2.imwrite(ouput_image_path, image)
    print(f"Predicted image saved at: {ouput_image_path}")


# Run the function on a test image
predict_and_draw(imagePath, model, class_names)



