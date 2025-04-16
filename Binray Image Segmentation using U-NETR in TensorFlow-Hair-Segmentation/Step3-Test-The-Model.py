import os

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from patchify import patchify
from Step2TrainUnetRModel import load_dataset, create_dir
from metrics import dice_loss

""" UNETR  Configration """
cf = {}
cf["image_size"] = 256
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

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing predictions """
    resultsFolder = "D:/Temp/Models/Unet-Binray/results"
    create_dir(resultsFolder)

    """ Load the model """
    model_path = os.path.join("D:/Temp/Models/Unet-Binray", "model.keras")
    model = tf.keras.models.load_model(model_path, custom_objects={"dice_loss": dice_loss})

    """ Dataset """
    dataset_path = "D:/Data-Sets-Object-Segmentation/figaro_hair_segmentation_1000"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)

    print(f"Train: \t{len(train_x)} - {len(train_y)}")
    print(f"Valid: \t{len(valid_x)} - {len(valid_y)}")
    print(f"Test: \t{len(test_x)} - {len(test_y)}")

    """ Prediction on the Test images """
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        """ Extracting the name with the full path """
        name = os.path.basename(x)
        print(name)

        """ Reading the image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (cf["image_size"], cf["image_size"]))
        x = image / 255.0

        patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])
        patches = patchify(x, patch_shape, cf["patch_size"])
        patches = np.reshape(patches, cf["flat_patches_shape"])
        patches = patches.astype(np.float32)
        patches = np.expand_dims(patches, axis=0)

        """ Read Mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (cf["image_size"], cf["image_size"]))
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=-1)
        mask = np.concatenate([mask, mask, mask], axis=-1)

        """ Prediction """
        pred = model.predict(patches, verbose=0)[0]
        pred = np.concatenate([pred, pred, pred], axis=-1)
        # pred = (pred > 0.5).astype(np.int32)

        """ Save final mask """
        line = np.ones((cf["image_size"], 10, 3)) * 255
        cat_images = np.concatenate([image, line, mask*255, line, pred*255], axis=1)
        save_image_path = os.path.join(resultsFolder,  name)
        #print(save_image_path)
        cv2.imwrite(save_image_path, cat_images)
        
        # OpenCV expects the input image to have a data type of uint8 or float32, but the cat_images variable has the type float64 (CV_64F)
        # lets convert it :
        cat_images_for_display = cat_images.astype(np.uint8)

        cv2.imshow("Result", cv2.cvtColor(cat_images_for_display, cv2.COLOR_RGB2BGR) )
        cv2.waitKey(1)  # Wait for a key press to proceed (use a duration in ms for automatic display)
    
cv2.destroyAllWindows()
