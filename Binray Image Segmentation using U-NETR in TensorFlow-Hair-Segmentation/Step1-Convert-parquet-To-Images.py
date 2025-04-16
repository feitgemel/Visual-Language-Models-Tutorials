# Dataset in parquet : https://huggingface.co/datasets/Allison/figaro_hair_segmentation_1000

# Convert the Parquet to images 
# =============================

import pandas as pd
import os
from PIL import Image
import io
from tqdm import tqdm

def extract_images_from_parquet(parquet_file, output_base_folder, dataset_type):
    """
    Extract images and labels from a Parquet file and save them to disk.

    Args:
        parquet_file (str): Path to the Parquet file.
        output_base_folder (str): Base folder to save the images.
        dataset_type (str): Dataset type ('train' or 'test').
    """
    # Load the Parquet file
    df = pd.read_parquet(parquet_file)
    
    # Create output folder structure
    dataset_folder = os.path.join(output_base_folder, dataset_type)
    image_folder = os.path.join(dataset_folder, "images")
    label_folder = os.path.join(dataset_folder, "masks")
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)

    # Save images and labels with tqdm progress bar
    print(f"Processing {dataset_type} dataset...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Saving {dataset_type} images"):
        # Process image column
        image_data = row['image']
        if isinstance(image_data, dict) and 'bytes' in image_data:
            image = Image.open(io.BytesIO(image_data['bytes']))
            image.save(os.path.join(image_folder, f"image_{idx}.png"))
        
        # Process label column
        label_data = row['label']
        if isinstance(label_data, dict) and 'bytes' in label_data:
            label = Image.open(io.BytesIO(label_data['bytes']))
            label.save(os.path.join(label_folder, f"label_{idx}.png"))

    print(f"{dataset_type.capitalize()} images and labels saved successfully.")

# Input Parquet files
train_parquet_file = "D:/Data-Sets-Object-Segmentation/figaro_hair_segmentation_1000/train-00000-of-00001-910d2af14081f419.parquet"
test_parquet_file = "D:/Data-Sets-Object-Segmentation/figaro_hair_segmentation_1000/validation-00000-of-00001-55044d1c657fc998.parquet"

# Output base folder
output_folder = "D:/Data-Sets-Object-Segmentation/figaro_hair_segmentation_1000"

# Extract images and labels for train and test datasets
extract_images_from_parquet(train_parquet_file, output_folder, "train")
extract_images_from_parquet(test_parquet_file, output_folder, "test")
