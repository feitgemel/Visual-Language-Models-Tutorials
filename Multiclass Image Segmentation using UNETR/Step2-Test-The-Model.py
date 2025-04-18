import os

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from patchify import patchify
from Step1TrainModel import load_dataset, create_dir
import matplotlib.pyplot as plt

""" UNETR  Configration """
cf = {}
cf["image_size"] = 256
cf["num_classes"] = 11
cf["num_channels"] = 3
cf["num_layers"] = 12
cf["hidden_dim"] = 128
cf["mlp_dim"] = 32
cf["num_heads"] = 6
cf["dropout_rate"] = 0.1
cf["patch_size"] = 16
cf["num_patches"] = (cf["image_size"]**2)//(cf["patch_size"]**2)
cf["flat_patches_shape"] = (
    cf["num_patches"],
    cf["patch_size"]*cf["patch_size"]*cf["num_channels"]
)

def grayscale_to_rgb(mask, rgb_codes):
    h, w = mask.shape[0], mask.shape[1]
    mask = mask.astype(np.int32)
    output = []

    for i, pixel in enumerate(mask.flatten()):
        output.append(rgb_codes[pixel])

    output = np.reshape(output, (h, w, 3))
    return output

def save_results(image_x, mask, pred, save_image_path):
    mask = np.expand_dims(mask, axis=-1)
    mask = grayscale_to_rgb(mask, rgb_codes)

    pred = np.expand_dims(pred, axis=-1)
    pred = grayscale_to_rgb(pred, rgb_codes)

    line = np.ones((image_x.shape[0], 10, 3)) * 255

    cat_images = np.concatenate([image_x, line, mask, line, pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)
    
    # OpenCV expects the input image to have a data type of uint8 or float32, but the cat_images variable has the type float64 (CV_64F)
    # lets convert it :
    cat_images_for_display = cat_images.astype(np.uint8)
    cv2.imshow("Result", cat_images_for_display )
    cv2.waitKey(1)  # Wait for a key press to proceed (use a duration in ms for automatic display)

if __name__ == "__main__":
    
    """ Directory for storing files """
    resultsFolder = "D:/Temp/Models/Unet-MultiClass/results"
    create_dir(resultsFolder)

    """ Load the model """
    model_path = os.path.join("D:/Temp/Models/Unet-MultiClass", "model.keras")
    model = tf.keras.models.load_model(model_path)

    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ RGB Code and Classes """
    rgb_codes = [
        [0, 0, 0], [0, 153, 255], [102, 255, 153], [0, 204, 153],
        [255, 255, 102], [255, 255, 204], [255, 153, 0], [255, 102, 255],
        [102, 0, 51], [255, 204, 255], [255, 0, 102]
    ]

    classes = [
        "background", "skin", "left eyebrow", "right eyebrow",
        "left eye", "right eye", "nose", "upper lip", "inner mouth",
        "lower lip", "hair"
    ]



    # Test and predict one image
    imgPath = "Visual-Language-Models-Tutorials/Multiclass Image Segmentation using UNETR/Eran Feit.jpg"
    image = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (cf["image_size"], cf["image_size"]))
    image_normelize = image / 255.0

    patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])
    img_to_patches = patchify(image_normelize, patch_shape, cf["patch_size"])
    img_to_patches = np.reshape(img_to_patches, cf["flat_patches_shape"])
    img_to_patches = img_to_patches.astype(np.float32) #[...]
    img_to_patches = np.expand_dims(img_to_patches, axis=0) # [1, ...]

    pred = model.predict(img_to_patches, verbose=0)[0]
    pred = np.argmax(pred, axis=-1) ## [0.1, 0.2, 0.1, 0.6] -> 3
    pred = pred.astype(np.int32)

    pred = np.expand_dims(pred, axis=-1)
    pred = grayscale_to_rgb(pred, rgb_codes)

    # Display the original image and the prediction side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Display the original image
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for matplotlib
    axes[0].axis('off')
    axes[0].set_title("Original Image")

    # Display the predicted image
    axes[1].imshow(pred)  # Prediction is already RGB
    axes[1].axis('off')
    axes[1].set_title("Predicted Segmentation")

    plt.savefig('Visual-Language-Models-Tutorials/Multiclass Image Segmentation using UNETR/Eran Feit-result.png')

    plt.tight_layout()
    plt.show()
    
    # Run a loop to all test images folder
    # ====================================


    """ Dataset """
    dataset_path = "D:/Data-Sets-Object-Segmentation/LaPa"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)

    print(f"Train: \t{len(train_x)} - {len(train_y)}")
    print(f"Valid: \t{len(valid_x)} - {len(valid_y)}")
    print(f"Test: \t{len(test_x)} - {len(test_y)}")

    """ Prediction """
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        """ Extracting the name """
        name = os.path.basename(x)
        #print(name)
        #name = os.path.basename(x).split(".")[0]


        """ Reading the image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (cf["image_size"], cf["image_size"]))
        x = image / 255.0

        patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])
        patches = patchify(x, patch_shape, cf["patch_size"])
        patches = np.reshape(patches, cf["flat_patches_shape"])
        patches = patches.astype(np.float32) #[...]
        patches = np.expand_dims(patches, axis=0) # [1, ...]

        """ Read Mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (cf["image_size"], cf["image_size"]))
        mask = mask.astype(np.int32)

        """ Prediction """
        pred = model.predict(patches, verbose=0)[0]
        pred = np.argmax(pred, axis=-1) ## [0.1, 0.2, 0.1, 0.6] -> 3
        pred = pred.astype(np.int32)

        """ Save the results """
        save_image_path = os.path.join(resultsFolder, name)
        save_results(image, mask, pred, save_image_path)

cv2.destroyAllWindows()