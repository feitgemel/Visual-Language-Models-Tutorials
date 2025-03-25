import torch 
from autodistill.detection import CaptionOntology
from autodistill_owlv2 import OWLv2 
import cv2 
import numpy as np
import matplotlib.pyplot as plt

# Define an ontology to map class names to our OWLvit prompt 

base_model = OWLv2(
    ontology=CaptionOntology(
        {
            "a basketball": "ball",
            "a tree": "tree"
        }
)
)    

# image path 
image_path = "Visual-Language-Models-Tutorials/Auto Label Custom Images using Transformer Owlv2/Basketball.jpg" 

# load image 
original_image = cv2.imread(image_path)
image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) # convert BGR to RGB

# Run inference 
results = base_model.predict(image_path)
print(results)


detections = {
    "xyxy": results.xyxy, # Bounding box coordinates
    "confidence": results.confidence, # Confidence score
    "class_id" : results.class_id, # Class ID
}

# Class mapping (adjust based on ontology)
class_mapping = {
    0: "basketball",
    1: "tree"
}

# Create a copy of the original image for draw annotations
annotated_image = image.copy()


# Iterate through the detections and draw bounding boxes and lables
for box , confidence , class_id in zip(detections["xyxy"], detections["confidence"], detections["class_id"]):
    x_min , y_min , x_max , y_max = map(int, box)
    label = f"{class_mapping[class_id]}: {confidence:.2f}"
    color = (255,0,0) if class_id == 0 else (0,255,0) # Blue for basket ball , Green for tree


    # Draw bounding box
    cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), color, thickness=6)

    # add the label above the bounding box
    cv2.putText(annotated_image, label, (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
   

# Display the image
plt.figure(figsize=(20,10))

# Original image 
plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

#Annotated image
plt.subplot(1,2,2)
plt.imshow(annotated_image)
plt.title("Annotated Image")
plt.axis("off")

# Show
plt.show()
