# Dataset : https://www.kaggle.com/datasets/adamridene/star-wars-characters

import os
import shutil
import random

def create_folders(base_path, categories):
    for category in categories:
        os.makedirs(os.path.join(base_path, 'Train', category), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'Val', category), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'Test', category), exist_ok=True)

def split_data(source_folder, dest_folder, train_ratio=0.7, validate_ratio=0.2):
    categories = [d for d in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, d))]
    create_folders(dest_folder, categories)
    
    for category in categories:
        category_path = os.path.join(source_folder, category)
        images = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
        random.shuffle(images)
        
        train_split = int(len(images) * train_ratio)
        validate_split = int(len(images) * (train_ratio + validate_ratio))
        
        train_images = images[:train_split]
        validate_images = images[train_split:validate_split]
        test_images = images[validate_split:]
        
        for image in train_images:
            shutil.copy(os.path.join(category_path, image), os.path.join(dest_folder, 'Train', category, image))
        
        for image in validate_images:
            shutil.copy(os.path.join(category_path, image), os.path.join(dest_folder, 'Val', category, image))
        
        for image in test_images:
            shutil.copy(os.path.join(category_path, image), os.path.join(dest_folder, 'Test', category, image))

source_folder = 'D:/Data-Sets-Image-Classification/Star-Wars-Characters'
dest_folder = 'D:/Data-Sets-Image-Classification/Star-Wars-Characters-For-Classification'
split_data(source_folder, dest_folder)

