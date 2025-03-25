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


# Class mapping (adjust based on ontology)
class_mapping = {
    0: "basketball",
    1: "tree"
}


input_path = "Visual-Language-Models-Tutorials/Auto Label Custom Images using Transformer Owlv2/sample-images"
output_path = "Visual-Language-Models-Tutorials/Auto Label Custom Images using Transformer Owlv2/output"

base_model.label(input_folder=input_path, output_folder=output_path, extension=".jpg")
