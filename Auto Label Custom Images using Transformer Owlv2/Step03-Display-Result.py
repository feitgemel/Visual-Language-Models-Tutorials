import os
import random
import cv2
import matplotlib.pyplot as plt

# Define directories for images and YOLO annotations
image_dir = "Visual-Language-Models-Tutorials/Auto Label Custom Images using Transformer Owlv2/output/train/images"
annotation_dir = "Visual-Language-Models-Tutorials/Auto Label Custom Images using Transformer Owlv2/output/train/labels"

# Get all image files (assuming jpg and png formats)
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

# Select 4 random images
selected_images = random.sample(image_files, 4)

# Function to read YOLO annotations and parse bounding boxes
def read_yolo_annotations(annot_path, img_width, img_height):
    boxes = []
    if os.path.exists(annot_path):
        with open(annot_path, "r") as file:
            for line in file:
                values = line.strip().split()
                class_id = int(values[0])  # First value is class ID
                x_center, y_center, width, height = map(float, values[1:])

                # Convert YOLO format to pixel coordinates
                x1 = int((x_center - width / 2) * img_width)
                y1 = int((y_center - height / 2) * img_height)
                x2 = int((x_center + width / 2) * img_width)
                y2 = int((y_center + height / 2) * img_height)

                boxes.append((class_id, x1, y1, x2, y2))
    return boxes

# Display images and annotations
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for ax, img_name in zip(axes.flatten(), selected_images):
    img_path = os.path.join(image_dir, img_name)
    annot_path = os.path.join(annotation_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
    
    # Read image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width, _ = img.shape
    
    # Read annotations
    boxes = read_yolo_annotations(annot_path, img_width, img_height)
    
    # Draw bounding boxes
    for class_id, x1, y1, x2, y2 in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 5)  # Blue bounding box
        cv2.putText(img, str(class_id), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 0, 0), 2, cv2.LINE_AA)  # Class label

    # Display image
    ax.imshow(img)
    ax.set_title(f"Image: {img_name}")
    ax.axis("off")

plt.tight_layout()
plt.show()
