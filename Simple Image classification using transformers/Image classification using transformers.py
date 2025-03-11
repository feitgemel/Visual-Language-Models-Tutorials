import cv2 

img_path = "Visual-Language-Models-Tutorials/Simple Image classification using transformers/Basketball.jpg"

# load and resize image  
img = cv2.imread(img_path)
scale_percent = 20 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

# convert image to RGB format (required for VITImagepreprocessing)
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

from transformers import ViTImageProcessor, ViTForImageClassification

# load the image processor and model
image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# process the image and extract the features
inputs = image_processor(images=rgb_img , return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits

# Get the predicted class
preidcted_class_idx = logits.argmax(-1).item()
print(preidcted_class_idx)

predicted_label = model.config.id2label[preidcted_class_idx]
print(predicted_label)

# display the predicted class on the image 
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0, 255, 0)
thickness = 2

cv2.putText(img, predicted_label, (50, 50), font, fontScale, fontColor, thickness, cv2.LINE_AA)

cv2.imwrite('d:/temp/output.jpg', img)

# display image
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

