""" 
Python script to perform auto-evaluation for object-addition, object-replacement, alter-parts, positional-addition, position-replacement, size

Requirements:
- The user needs to upload the original image and the generated image after edit, along with the information about edit-dimension from EditVal dataset

"""
# Libraries
import torch 
from PIL import Image
import open_clip
import torchvision 
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.datasets  import VisionDataset 
from typing import Any, Callable, List, Optional, Tuple
import os 
import random 
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
import numpy as np 
import torch.nn.functional as F
import argparse
from omegaconf import OmegaConf
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam, AdamW 
from tqdm import tqdm
from torch import nn
import json 
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

# Accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from torch.utils.data import Dataset
import itertools 
import PIL 
import math 

import requests
from PIL import Image
import torch

from transformers import OwlViTProcessor, OwlViTForObjectDetection
import json 
import matplotlib.pyplot as plt 
import pickle 

# Evaluating object-addition
def evaluate_object_addition(args):
    # 
    print(f'###### Evaluating for Object-Addition #####')

    # OwL-ViT Processor
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    # OwlViT Model
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    # Open the edited image
    image = Image.open(args.edited_path)
    # Object which is added
    object_ = args.to 
    # Original Object
    class_curr = args.class_ 
    texts = [["a photo of a " + str(object_), "a photo of a " + str(class_curr)]] # e.g., [["a photo of a bread", "a photo of a cup", "a photo of a finger"]]
    # Processor
    inputs = processor(text=texts, images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)


    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    # Original presence
    orig = 0

    # New Object presence 
    addition = 0

    # Score threshold (Default from OwL-ViT paper)
    score_threshold = 0.1
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        if score >= score_threshold:
            if label == 0:
                addition += 1
            
            if label == 1:
                orig += 1

    
    # If original object is present and new object is present -- then score is 1 
    if orig > 0 and addition > 0:
        return 1 


    return 0

# Evaluating object-replacement
def evaluate_object_replacement(args):
    print(f'###### Evaluating for Object-Replacement #####')

    # OwL-ViT Processor
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    # OwlViT Model
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    # Open the edited image
    image = Image.open(args.edited_path)
    # Object which is added
    object_ = args.to 
    # Original Object
    class_curr = args.class_ 
    texts = [["a photo of a " + str(object_), "a photo of a " + str(class_curr)]] # e.g., [["a photo of a bread", "a photo of a cup", "a photo of a finger"]]
    # Processor
    inputs = processor(text=texts, images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)


    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    # Original presence
    orig = 0

    # New Object presence 
    addition = 0

    # Score threshold (Default from OwL-ViT paper)
    score_threshold = 0.1
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        if score >= score_threshold:
            #print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
            if label == 0:
                addition += 1
            
            if label == 1:
                orig += 1

    
    # If original object is NOT present and new object is present -- then score is 1
    if orig == 0 and addition > 0:
        return 1 

    return 0

# Evaluating alter-parts
def evaluate_alter_parts(args):
    print(f'###### Evaluating for Alter-Parts #####')

    # OwL-ViT Processor
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    # OwlViT Model
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    # Open the edited image
    image = Image.open(args.edited_path)
    # Location attribute
    object_ = args.to 
    
    # Original Object
    class_curr = args.class_ 
    texts = [["a photo of a " + str(object_), "a photo of a " + str(class_curr)]] # e.g., [["a photo of a bread", "a photo of a cup", "a photo of a finger"]]
    # Processor
    inputs = processor(text=texts, images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    

    # Target sizes
    target_sizes = torch.Tensor([image.size[::-1]])
    
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    # Original
    orig = 0
    addition = 0
    # Score threshold 
    score_threshold = 0.1
    box_original = None 
    box_addition = None 
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        if score >= score_threshold:
            #print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
            if label == 0:
                addition += 1
                box_addition = box 
            
            if label == 1:
                orig += 1
                box_original = box 

    
    # If original and new object both are present
    if orig >0  and addition > 0:
        # Check the bounding box condition
        cx_addition, cy_addition, _, _ = box_addition
        cx_original, cy_original, w_original, h_original = box_original

        # Check if both objects are present adn the added object is inside the main objects bounding box location
        if cx_addition < (cx_original + w_original/2) and cx_addition>(cx_original - w_original/2) and cy_addition < (cy_original + h_original/2) and cy_addition > (cy_original-h_original/2):
            return 1
        
        else:
            return 0
    
    else:
        return 0


    # Return 0 if the object is not added OR if the object is added but the bounding box rules are not followed.
    return 0


# Evaluating positional-addition
def evaluate_positional_addition(args):
    print(f'###### Evaluating for Positional-Addition #####')

    # OwL-ViT Processor
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    # OwlViT Model
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    # Open the edited image
    image = Image.open(args.edited_path)
    # Location attribute
    to_attribute = args.to 
    # Object which needs to be added 
    object_ = to_attribute.split(' ')[0]
    
    # Location where the new object needs to be added
    pos = to_attribute.split(' ')[-1]

    # Original Object
    class_curr = args.class_ 
    texts = [["a photo of a " + str(object_), "a photo of a " + str(class_curr)]] # e.g., [["a photo of a bread", "a photo of a cup", "a photo of a finger"]]
    # Processor
    inputs = processor(text=texts, images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    

    # Target sizes
    target_sizes = torch.Tensor([image.size[::-1]])
            
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    # Original
    orig = 0
    addition = 0
    # Score threshold 
    score_threshold = 0.1
    box_addition = None 
    box_original = None 
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        if score >= score_threshold:
            #print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
            if label == 0:
                addition += 1
                box_addition = box 
            if label == 1:
                orig += 1
                box_original = box 

    
    # If original object is present and new added object is present
    if orig > 0 and addition > 0:
        # Both objects are present and their positional understanding needs to be checked
        cx_addition, cy_addition, w_addition, h_addition = box_addition
        cx_original, cy_original, w_original, h_original = box_original

        # Bounding box coordinates of the added object
        addition_coordinates_x = [cx_addition-w_addition/2, cx_addition+w_addition/2, cx_addition+w_addition/2, cx_addition-w_addition/2]  
        addition_coordinates_y = [cy_addition-h_addition/2, cy_addition-h_addition/2, cy_addition+h_addition/2, cy_addition+h_addition/2]

        # New object coordinates
        mean_x_addition = sum(addition_coordinates_x)/len(addition_coordinates_x)
        mean_y_addition = sum(addition_coordinates_y)/len(addition_coordinates_y)

        # Bounding box coordinates of the original object
        original_coordinates_x = [cx_original-w_original/2, cx_original+w_original/2, cx_original+w_original/2, cx_original-w_original/2]  
        original_coordinates_y = [cy_original-h_original/2, cy_original-h_original/2, cy_original+h_original/2, cy_original+h_original/2]

        # Original object coordinates
        mean_x_original = sum(original_coordinates_x)/len(original_coordinates_x)
        mean_y_original = sum(original_coordinates_y)/len(original_coordinates_y)


        # Below
        if pos == 'below':
            if mean_y_addition < mean_y_original:
                return 1      
            
            else:
                return 0
        
        # Top
        elif pos == 'top':
            if mean_y_addition > mean_y_original:
                return 1
            
            else:
                return 0

        # Left
        elif pos == 'left':
            if mean_x_addition < mean_x_original:
                return 1
            
            else:
                return 0
        
        # Right
        elif pos == 'right':
            if mean_x_addition > mean_x_original:
                return 1
            
            else:
                return 0


    # Return 0 if none of the conditions are met 
    return 0

# Evaluating position-replacement
def position_replacement(args):
    print(f'###### Evaluating for Position-Replacement #####')

    # OwL-ViT Processor
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    # OwlViT Model
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    # Define delta for position-replacement
    delta = 200

    # Open the original image
    image = Image.open(args.orig_path)

    # Current Class
    class_curr = args.class_
    texts = [["a photo of a " + str(class_curr)]] #[["a photo of a bread", "a photo of a cup", "a photo of a finger"]]
    
    # Inputs
    inputs = processor(text=texts, images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)


    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    # Original
    #orig = 0
    addition = 0
    # Score threshold 
    score_threshold = 0.1
    box_addition = None 
    #box_original = None 
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        if score >= score_threshold:
            #print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
            if label == 0:
                addition += 1
                box_addition = box 
    


    # Edited Path
    image = Image.open(args.edited_path) 
    texts = [["a photo of a " + str(class_curr)]] #[["a photo of a bread", "a photo of a cup", "a photo of a finger"]]
    
    inputs = processor(text=texts, images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    #print(texts)
    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])

    # Convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    # Original
    orig = 0
    #addition = 0
    orig_pos = None
    # Score threshold 
    score_threshold = 0.1
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        if score >= score_threshold:
            #print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
            if label == 0:
                orig += 1
                orig_pos = box 
    

    # If the object is present in both the original and the edited image 
    if orig > 0 and addition > 0:
        # Old image position
        old_image_pos = box_addition
        edited_image_pos = orig_pos 

        # Old
        cx_old, cy_old, _, _  = old_image_pos 
        # Edited
        cx_edited, cy_edited, _, _ = edited_image_pos

        # If object needs to be moved to the left
        if args.to == 'left':
            if cx_edited<(cx_old - delta):
                return 1
            
        # If object needs to be moved to the right
        elif args.to == 'right':
            if cx_edited>(cx_old + delta):
                return 1
            

    return 0

# Evaluating Size
def evaluate_size(args):
    print(f'###### Evaluating for Size #####')

    # OwL-ViT Processor
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    # OwlViT Model
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    # Open the original image
    image = Image.open(args.orig_path)
    # Current class
    class_curr = args.class_ 
    texts = [["a photo of a " + str(class_curr)]] #[["a photo of a bread", "a photo of a cup", "a photo of a finger"]]
    #print(texts)
    inputs = processor(text=texts, images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    # Original
    orig = 0
    # Score threshold 
    score_threshold = 0.1
    box_original = None 

    # Save the original bounding box configuration
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        if score >= score_threshold:
            #print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
            if label == 0:
                orig += 1
                box_original = box  


    
    # Image
    image = Image.open(img_)
    texts = [["a photo of a " + str(class_curr)]] #[["a photo of a bread", "a photo of a cup", "a photo of a finger"]]
    #print(texts)
    inputs = processor(text=texts, images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    # Original
    addition = 0
    # Score threshold 
    score_threshold = 0.1
    box_addition = None 
    #box_original = None 
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        if score >= score_threshold:
            #print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
            if label == 0:
                addition += 1
                box_addition = box 


    # Original object is present
    if orig > 0: 
        # Check if the edited image contains the object or not 
        if addition > 0:
            # Check the bounding box area 
            cx_addition, cy_addition, w_addition, h_addition = box_addition
            cx_original, cy_original, w_original, h_original = box_original

            # Check the area of the bounding box
            addition_coordinates_x = [cx_addition-w_addition/2, cx_addition+w_addition/2, cx_addition+w_addition/2, cx_addition-w_addition/2]  
            addition_coordinates_y = [cy_addition-h_addition/2, cy_addition-h_addition/2, cy_addition+h_addition/2, cy_addition+h_addition/2]

            # Check the area of the bounding box - 
            original_coordinates_x = [cx_original-w_original/2, cx_original+w_original/2, cx_original+w_original/2, cx_original-w_original/2]  
            original_coordinates_y = [cy_original-h_original/2, cy_original-h_original/2, cy_original+h_original/2, cy_original+h_original/2]

            # Addition Area
            area_addition = abs(addition_coordinates_x[0] - addition_coordinates_x[1]) * abs(addition_coordinates_y[0] - addition_coordinates_y[2])
            # Original Area
            area_original = abs(original_coordinates_x[0] - original_coordinates_x[1]) * abs(original_coordinates_y[0] - original_coordinates_y[2])

            # If the size of the object needs to be increased
            if args.to == 'large':
                if area_addition > area_original:
                    return 1
            
            # If the size of the object needs to be decreased
            elif args.to == 'small':
                if area_addition < area_original:
                    return 1 
    

    # If none of the conditions met, then return 0
    return 0



# Main function which orchestrates the edits
def main():
    # Argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_", default='sink', type=str, required=False, help="Class Label")
    
    # Options corresponding to the attributes of 
    parser.add_argument("--edit_model", default='pix2pix', type=str, required=False, help="Diffusion Model to use for editing")
    parser.add_argument("--edit_dimension", default='object_addition', type=str, required=False, help="Diffusion Model Editing Dimension to Use for Editing")
    parser.add_argument("--from", default='None', type=str, required=False, help="Diffusion Model Editing Dimension to Use for Editing")
    parser.add_argument("--to", default='None', type=str, required=False, help="Diffusion Model Editing Dimension to Use for Editing")

    # Original Image
    parser.add_argument("--orig_path", default='/images', type=str, required=False, help="Diffusion Model Editing Dimension to Use for Editing")
    # Path for the edited Image
    parser.add_argument("--edited_path", default='/images', type=str, required=False, help="Diffusion Model Editing Dimension to Use for Editing")

    # Argument parser
    args = parser.parse_args()

    # Evaluation for object-addition
    if args.edit_dimension == 'object_addition':
        score = evaluate_object_addition(args)
    
    # Evaluation for object-replacement
    elif args.edit_dimension == 'object_replacement':
        score = evaluate_object_replacement(args)
    
    # Evaluation for alter-parts
    elif args.edit_dimension == 'alter_parts':
        score = evaluate_alter_parts(args)
    
    # Evaluation for positional-addition
    elif args.edit_dimension == 'positional-addition':
        score = evaluate_positional_addition(args)
    
    # Evaluation for position-replacement
    elif args.edit_dimension == 'position_replacement':
        score = evaluate_position_replacement(args)
    
    # Evaluation for size
    elif args.edit_dimension == 'size':
        score = evaluate_size(args)

    print(f'The score for edit dimension : {args.edit_dimension} for the edited image is {score}')

    return 



# Main function
if __name__ == "__main__":
    # Main function
    main()

