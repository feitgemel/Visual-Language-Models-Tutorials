# Dataset in parquet : https://huggingface.co/datasets/Allison/figaro_hair_segmentation_1000

import os

import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from patchify import patchify
from unetr_2d import build_unetr_2d
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

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(path, split=0.1):
    """ Loading the list of images and masks """
    X = sorted(glob(os.path.join(path, "train", "images", "*.png")))
    Y = sorted(glob(os.path.join(path, "train", "masks", "*.png")))

    """ Spliting the data into training and testing """
    split_size = int(len(X) * split)

    train_x, valid_x = train_test_split(X, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(Y, test_size=split_size, random_state=42)

    # relevant if you dont have test data 
    #train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
    #train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=42)

    # get the test data 
    test_x = sorted(glob(os.path.join(path, "test", "images", "*.png")))
    test_y = sorted(glob(os.path.join(path, "test", "masks", "*.png")))

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

# Create pataches to every image
def read_image(path):
    path = path.decode()
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (cf["image_size"], cf["image_size"]))
    image = image / 255.0 # normalize between 0 to 1

    """ Processing to patches """
    patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])
    patches = patchify(image, patch_shape, cf["patch_size"])
    patches = np.reshape(patches, cf["flat_patches_shape"])
    patches = patches.astype(np.float32)

    return patches

# each mask contain two values : 0 and 255 . 0 is the background and 255 is the human hair
# we change it to 0 and 1 

def read_mask(path):
    path = path.decode()
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (cf["image_size"], cf["image_size"]))
    mask = mask / 255.0
    mask = mask.astype(np.float32)
    mask = np.expand_dims(mask, axis=-1) 
    return mask


# convert it to pathces
def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape(cf["flat_patches_shape"])
    y.set_shape([cf["image_size"], cf["image_size"], 1])
    return x, y

# read , resize and create the dataset pipeline 
def tf_dataset(X, Y, batch=2):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.map(tf_parse).batch(batch).prefetch(10)
    return ds

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("D:/Temp/Models/Unet-Binray")

    """ Hyperparameters """
    batch_size = 8
    lr = 0.1
    num_epochs = 500
    model_path = os.path.join("D:/Temp/Models/Unet-Binray", "model.keras")
    csv_path = os.path.join("D:/Temp/Models/Unet-Binray", "log.csv")

    """ Dataset """
    dataset_path = "D:/Data-Sets-Object-Segmentation/figaro_hair_segmentation_1000"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)

    # Here we call and run the dataset Pipline functions
    # The parameters are the list of images files and list of masks files
    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

    """ LOAD Model """
    model = build_unetr_2d(cf)
    model.compile(loss=dice_loss, optimizer=SGD(lr))
    print(model.summary())

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
    ]

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks
    )