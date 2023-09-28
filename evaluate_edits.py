""" 
Script for the evaluation workflow of the edited images from various diffusion-based editing frameworks;

"""

# Libraries
import torch 
from PIL import Image
import open_clip
import torchvision 
#import torchvision.datasets as dset
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

# The images are saved in the following way;
''' method_name
	→ attribute_name
		→ image_name
			→ image_name_unedited.png
→ image_name_current_attribute_local_img_guide_1_text_guide_1.png '''


data_path = './edit_bench_data/'



# Function to evaluate for positional replacement
def positional_replacement(edit_model):
    print(f'Attribute: Positional Replacement')

    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32", cache_dir='./cache')
    # OwlViT Model
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32", cache_dir = './cache')

    ############################## Put constraints based on the condition ##############################
    orig_path = data_path
    # Pix2pix
    if edit_model == 'pix2pix':
        print(f'Evaluating for Positional Replacement with pix2pix....')
        path_images = orig_path + edit_model
        path_to_extract = os.path.join(path_images, 'position_replacement')

    # Textual Inversion
    elif edit_model == 'textual_inversion':
        print(f'Evaluating for Positional Replacement with textual inversion .... ')
        path_images = orig_path + edit_model 
        path_to_extract = os.path.join(path_images, 'position_replacement')
    
    # SDE-Edit
    elif edit_model == 'sde_edit':
        print(f'Evaluating for Positional Replacement with SDE-edit ......')
        path_images = os.path.join(orig_path, edit_model)
        path_to_extract = os.path.join(path_images, 'position_replacement')
    
    # Null-Text Inversion
    elif edit_model == 'null_text':
        print(f'Evaluating for Positional Replacement with Null Text ......')
        path_images = os.path.join(orig_path, 'null_text_renamed')
        path_to_extract = os.path.join(path_images, 'position_replacement')

    # Imagic
    elif edit_model == 'imagic':
        print(f'Evaluating for Positional Replacement with Imagic ......')
        path_images = s.path.join(orig_path, edit_model)
        path_to_extract = os.path.join(path_images, 'position_replacement')

    # Path to extract
    print(f'############# Path to extract ###########: {path_to_extract}')
    all_dirs = os.listdir(path_to_extract)
    total_images = len(all_dirs)

    # Total paths for all the images 
    paths_all = [os.path.join(path_to_extract, a) for a in all_dirs]
    

    ##################################### Results Saving ###############################
    result_path = os.path.join(data_path, 'results', edit_model)
    
    # Creata path if it doesn't exist
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        print(f'Path created: {result_path}')

    # Saving for all the dictionary
    result_dict = {}
    #####################################################################################

    ########################## Pix2pix scales ##########################
    # Higher value will ensure faithfulness to the original image
    image_guidance_scales = [1, 1.5, 2.0, 2.5]

    # Higher guidance scale will ensure faithfulness to the text, with a drawback of reducing the fidelity of the image
    text_guidance_scale = [6.5, 7.0, 7.5, 8.0, 8.5, 9.5]

    # SDE-Edit noise scales
    sde_edit_noise_scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    ############# Define the storage for the correct edits ###########

    ##################### Pix2pix ####################
    pix2pix_total = {}
    pix2pix_correct = {}

    # Populate
    for img_guide in image_guidance_scales:
        for txt_guide in text_guidance_scale:
            key_curr = str(img_guide) + '_' + str(txt_guide)
            pix2pix_total[key_curr] = 0
            pix2pix_correct[key_curr] = 0
    

   
    ######################## SDE edit ####################
    sde_edit_total = {}
    sde_edit_correct = {} 
    
    for noise in sde_edit_noise_scales:
        sde_edit_total[str(noise)] = 0
        sde_edit_correct[str(noise)] = 0
    
    
    ################# Textual Inversion ###############
    correct_edit = 0
    total_edits = 0 

    ###########################################################


    # 
    # Iterate for every image and extract the edited images for the condition
    """ 
        Step 1: Iterate through each image -- it will have all the edits corresponding to it 
        Step 2: Inside each for loop, iterate through all the edits corresponding to the image
    """
    for path_ in paths_all:
        # Get all files 
        image_paths = os.listdir(path_)
        total_paths = [os.path.join(path_, a) for a in image_paths]
        print(len(total_paths))
        #print(total_paths)
        orig_image = [a for a in total_paths if 'unedited' in a][0]
        paths_without_orig = [a for a in total_paths if 'unedited' not in a]
        


        ################################ Obtaining the class of the given image_id ########################
        # Open
        f = open('object_new.json')
        # Load the data
        edit_data = json.load(f)
        print(f'Paths ... : {path_}')
        image_id_curr = path_.split('/')[-1]
        print(f'Current Image ID: {image_id_curr}')

        classes = list(edit_data.keys())
        class_curr = None 
        edit_instructions = []

        for cls_ in classes:
            curr_tuple = list(edit_data[cls_].keys())

            # Image_id
            if image_id_curr in curr_tuple:
                # Search the instructions from the current tuple 
                class_curr = cls_ 

        #######################################################################################################

        # Accuracy for the current edit corresponding to the image
        #correct_edit = 0
        # Total number of edits for the given image
        #total_edits = 0

        # Paths without original 
        #print(paths_without_orig)
        image_paths_without_orig = paths_without_orig

        # Total number of edits to evaluate
        print(f'Total Number of Edits to Evaluate for Object Addition: {len(image_paths_without_orig)}')
        # TODO --- Separate out the accuracies for each image
        # Iterate through all the edits for the particular image id
        for img_ in image_paths_without_orig:
            # Pix2Pix
            if edit_model == 'pix2pix':
                # Image tag
                image_tag = img_.split('/')[-1].split('_')
                pos_ = image_tag[3]
                
                # Image-tag
                img_guide = image_tag[4]
                text_guide = image_tag[5][:-4]
            
            # SDE_edit
            elif edit_model == 'sde_edit':
                # Image-Tag
                image_tag = img_.split('/')[-1].split('_')
                pos_ = image_tag[3]
                noise_guide = image_tag[4][:-4]
                key_curr = str(noise_guide)


            # Textual Inversion
            elif edit_model == 'textual_inversion':
                image_tag = img_.split('/')[-1].split('_')
                pos_ = image_tag[3]
            
            # Null-Text
            elif edit_model == 'null_text':
                image_tag = img_.split('/')[-1].split('_')
                pos_ = image_tag[1]
            
            # Imagic
            elif edit_model == 'imagic':
                image_tag = img_.split('/')[-1].split('_')
                pos_ = image_tag[1]

            print(f'########## Position: {pos_} ##############')

            ################################## OLD image #############################
            image = Image.open(orig_image)
            texts = [["a photo of a " + str(class_curr)]] #[["a photo of a bread", "a photo of a cup", "a photo of a finger"]]
            #print(texts)
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



            ################################### New image #######################################
            # Dreambooth # 
            print(f'Class: {class_curr}')

            image = Image.open(img_)
            texts = [["a photo of a " + str(class_curr)]] #[["a photo of a bread", "a photo of a cup", "a photo of a finger"]]
            #print(texts)
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
        


            #  If original is present
            if orig > 0 and addition > 0:
                # original object i spresent
                old_image_pos = box_addition
                edited_image_pos = orig_pos 

                # Old
                cx_old, cy_old, _, _  = old_image_pos 
                # Edited
                cx_edited, cy_edited, _, _ = edited_image_pos

                # 
                if pos_ == 'left':
                    if cx_edited<cx_old and cy_edited<cy_old:
                        if edit_model == 'pix2pix':
                            pix2pix_correct[key_curr] += 1
                            result_dict[img_] = 1
                        
                        elif edit_model == 'textual_inversion':
                            correct_edit += 1
                            result_dict[img_] = 1
                        
                        elif edit_model == 'sde_edit':
                            sde_edit_correct[key_curr] += 1
                            result_dict[img_] = 1
                        
                        else:
                            correct_edit += 1
                            result_dict[img_] = 1
                    
                    else:
                        result_dict[img_] = 0


                elif pos_ == 'right':
                    # Condition Met
                    if cx_edited>cx_old and cy_edited>cy_old:
                        if edit_model == 'pix2pix':
                            pix2pix_correct[key_curr] += 1
                            result_dict[img_] = 1
                        
                        elif edit_model == 'textual_inversion':
                            correct_edit += 1
                            result_dict[img_] = 1
                        

                        elif edit_model == 'sde_edit':
                            sde_edit_correct[key_curr] += 1
                            result_dict[img_] = 1
                        
                        else:
                            correct_edit += 1
                            result_dict[img_] = 1
                    # Condition not met
                    else:
                        result_dict[img_] = 0

            # Original object is present and the original object is present in new one 
            else:
               result_dict[img_] = 0

            
            # Total Edits
            if edit_model == 'pix2pix':
                pix2pix_total[key_curr] += 1
            
            elif edit_model == 'textual_inversion':
                total_edits += 1
            
            elif edit_model == 'sde_edit':
                sde_edit_total[key_curr] += 1
            
            else:
                total_edits += 1

            
            

    # Save dictionary path
    save_dict_path = os.path.join(result_path, 'position_replacement.pkl')
    with open(save_dict_path, 'wb') as f:
        pickle.dump(result_dict, f)
        
    with open(save_dict_path, 'rb') as f:
        loaded_dict = pickle.load(f)
    

    # Load the edited image corresponding to the image 
    if edit_model == 'pix2pix':
        print(f'Correct: {pix2pix_correct}')
        print(f'Total: {pix2pix_total}')
    

    elif edit_model == 'sde_edit':
        print(f'Correct: {sde_edit_correct}')
        print(f'Total: {sde_edit_total}')
    
    else:
        print(f'Correct: {correct_edit}')
        print(f'Total: {total_edits}')


    return 


# Function to add objects
def object_addition(edit_model):
    print(f'Attribute: Object Addition')
    # Old object has to be present
    # New object has to be present 
    # Old objects position is not changed drastically 
    
    ############################# Load Object-Detector #############################
    # Processor ffor OwlViT
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32", cache_dir='./cache')
    # OwlViT Model
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32", cache_dir = './cache')

    ############################## Put constraints based on the condition ##############################
    orig_path = data_path
    # Pix2pix
    if edit_model == 'pix2pix':
        print(f'Evaluating for Object Addition with pix2pix....')
        path_images = orig_path + edit_model
        path_to_extract = os.path.join(path_images, 'object_addition')

    # Textual Inversion
    elif edit_model == 'textual_inversion':
        print(f'Evaluating for Object Addition with textual inversion .... ')
        path_images = orig_path + edit_model
        path_to_extract = os.path.join(path_images, 'object_addition')
    
    # SDE-Edit
    elif edit_model == 'sde_edit':
        print(f'Evaluating for Object Addition with SDE-edit ......')
        path_images = orig_path + edit_model
        path_to_extract = os.path.join(path_images, 'object_addition')
    
    # Null-text-inversion
    elif edit_model == 'null_text':
        print(f'Evaluating for Object Addition with Null-Text')
        path_images = orig_path + 'null_text_renamed'
        path_to_extract = os.path.join(path_images, 'object_addition')

    # Imagic 
    elif edit_model == 'imagic':
        print(f'Evaluating for Object Addition with Null-Text')
        path_images = orig_path + edit_model 
        path_to_extract = os.path.join(path_images, 'object_addition')

    # Path to extract
    print(f'############# Path to extract ###########: {path_to_extract}')
    all_dirs = os.listdir(path_to_extract)
    total_images = len(all_dirs)

    # Total paths for all the images 
    paths_all = [os.path.join(path_to_extract, a) for a in all_dirs]
    

    ############################## Create File corresponding to method and edit #####################
    result_path = os.path.join(data_path, 'results', edit_model)
    
    # Creata path if it doesn't exist
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        print(f'Path created: {result_path}')

    #print(f'Result Path: {result_path}')
    result_dict = {}
    ##################################################################################################



    ########################## Pix2pix scales ##########################
    # Higher value will ensure faithfulness to the original image
    image_guidance_scales = [1, 1.5, 2.0, 2.5]

    # Higher guidance scale will ensure faithfulness to the text, with a drawback of reducing the fidelity of the image
    text_guidance_scale = [6.5, 7.0, 7.5, 8.0, 8.5, 9.5]

    # SDE-Edit noise scales
    sde_edit_noise_scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    ############# Define the storage for the correct edits ###########

    ##################### Pix2pix ####################
    pix2pix_total = {}
    pix2pix_correct = {}

    # Populate
    for img_guide in image_guidance_scales:
        for txt_guide in text_guidance_scale:
            key_curr = str(img_guide) + '_' + str(txt_guide)
            pix2pix_total[key_curr] = 0
            pix2pix_correct[key_curr] = 0
    

   
    ######################## SDE edit ####################
    sde_edit_total = {}
    sde_edit_correct = {} 
    
    for noise in sde_edit_noise_scales:
        sde_edit_total[str(noise)] = 0
        sde_edit_correct[str(noise)] = 0
    
    
    ################# Textual Inversion ###############
    correct_edit = 0
    total_edits = 0 

    ###########################################################

    # Iterate for every image and extract the edited images for the condition
    """ 
        Step 1: Iterate through each image -- it will have all the edits corresponding to it 
        Step 2: Inside each for loop, iterate through all the edits corresponding to the image
    """
    for path_ in paths_all:
        # Get all files 
        image_paths = os.listdir(path_)
        total_paths = [os.path.join(path_, a) for a in image_paths]
        #print(len(total_paths))
        #print(total_paths)
        orig_image = [a for a in total_paths if 'unedited' in a][0]
        paths_without_orig = [a for a in total_paths if 'unedited' not in a]
        

        ################################ Obtaining the class of the given image_id ########################
        # Open
        f = open('object_new.json')
        # Load the data
        edit_data = json.load(f)
        print(f'Paths ... : {path_}')
        image_id_curr = path_.split('/')[-1]
        print(f'Current Image ID: {image_id_curr}')

        classes = list(edit_data.keys())
        class_curr = None 
        edit_instructions = []

        for cls_ in classes:
            curr_tuple = list(edit_data[cls_].keys())

            # Image_id
            if image_id_curr in curr_tuple:
                # Search the instructions from the current tuple 
                class_curr = cls_ 

        #######################################################################################################

        # Accuracy for the current edit corresponding to the image
        #correct_edit = 0
        # Total number of edits for the given image
        #total_edits = 0

        # Paths without original 
        #print(paths_without_orig)
        image_paths_without_orig = paths_without_orig

        # Total number of edits to evaluate
        print(f'Total Number of Edits to Evaluate for Object Addition: {len(image_paths_without_orig)}')
        # TODO --- Separate out the accuracies for each image
        # Iterate through all the edits for the particular image id
        for img_ in image_paths_without_orig:
            # Pix2Pix
            if edit_model == 'pix2pix':
                # Image tag
                image_tag = img_.split('/')[-1].split('_')
                object_ = image_tag[3]
                
                # Image-tag
                img_guide = image_tag[4]
                text_guide = image_tag[5][:-4]
                key_curr = str(img_guide) + '_' + str(text_guide)
            
            # SDE_edit
            elif edit_model == 'sde_edit':
                # Image-Tag
                image_tag = img_.split('/')[-1].split('_')
                object_ = image_tag[3]
                noise_guide = image_tag[4][:-4]
                key_curr = str(noise_guide)

                #print(f'Noise guide: {noise_guide}')

            # Textual Inversion
            elif edit_model == 'textual_inversion':
                image_tag = img_.split('/')[-1].split('_')
                object_ = image_tag[3]
                # No hyperparameter for this method
            

            # Null-Text Inversion 
            elif edit_model == 'null_text':
                image_tag = img_.split('/')[-1].split('_')
                object_ = image_tag[1]
                #print(f'Object to be added: {object_}')
            
            # SINE 
            elif edit_model == 'imagic':
                image_tag = img_.split('/')[-1].split('_')
                object_ = image_tag[1]
                print(f'Object to be added: {object_}')
            
            # TODO : Dreambooth, SINE, DiffEdit


            # URL
            #url = "http://images.cocodataset.org/val2017/000000039769.jpg"

            ######################################### Object detector part #######################################
            #image = Image.open(requests.get(url, stream=True).raw)
            image = Image.open(img_)
            texts = [["a photo of a " + str(object_), "a photo of a " + str(class_curr)]] #[["a photo of a bread", "a photo of a cup", "a photo of a finger"]]
            #print(texts)
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
            addition = 0
            # Score threshold 
            score_threshold = 0.1
            for box, score, label in zip(boxes, scores, labels):
                box = [round(i, 2) for i in box.tolist()]
                if score >= score_threshold:
                    #print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
                    if label == 0:
                        addition += 1
                    
                    if label == 1:
                        orig += 1

            
            # If original object is present and new object is present
            if orig > 0 and addition > 0:
                if edit_model == 'pix2pix':
                    pix2pix_correct[key_curr] += 1
                    result_dict[img_] = 1
                
                elif edit_model == 'textual_inversion':
                    correct_edit += 1
                    result_dict[img_] = 1
                
                elif edit_model == 'sde_edit':
                    sde_edit_correct[key_curr] += 1
                    result_dict[img_] = 1
                
                else:
                    correct_edit += 1
                    result_dict[img_] = 1

            # Condition not met
            else:
                result_dict[img_] = 0

            
            print(f'Image tag: {img_}')

            # Total Edits
            if edit_model == 'pix2pix':
                pix2pix_total[key_curr] += 1
            
            elif edit_model == 'textual_inversion':
                total_edits += 1
            
            elif edit_model == 'sde_edit':
                sde_edit_total[key_curr] += 1
            
            else:
                total_edits += 1


         
    
    # Save the result_dict in  result path 
    #print(result_path)
    save_dict_path = os.path.join(result_path, 'object_addition.pkl')
    with open(save_dict_path, 'wb') as f:
        pickle.dump(result_dict, f)
        
    with open(save_dict_path, 'rb') as f:
        loaded_dict = pickle.load(f)
    

    print(f'##### Loaded dict : {loaded_dict}')
    # Load the original image 
    
    # Load the edited image corresponding to the image 
    if edit_model == 'pix2pix':
        print(f'Correct: {pix2pix_correct}')
        print(f'Total: {pix2pix_total}')
        #print(f'Acc: {pix2pix_correct/pix2pix_total}')

    elif edit_model == 'sde_edit':
        print(f'Correct: {sde_edit_correct}')
        print(f'Total: {sde_edit_total}')
        #print(f'Acc: {sde_edit_correct/sde_edit_total}')
    else:
        print(f'Correct: {correct_edit}')
        print(f'Total: {total_edits}')
        #print(f'Acc: {correct_edits/total_edits}')
    


    return  

# Function for object deletion 
def object_deletion(edit_model):
    print(f'Attribute: Object Deletion')
    # Old object needs to be deleted

    ############################# Load Object-Detector #############################
    # Processor ffor OwlViT
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32", cache_dir='./cache')
    # OwlViT Model
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32", cache_dir = './cache')

    ############################## Put constraints based on the condition ##############################
    # Pix2pix
    if edit_model == 'pix2pix':
        print(f'Evaluating for Object Addition with pix2pix....')
        path_images = data_path + edit_model
        path_to_extract = os.path.join(path_images, 'object_replacement')

    # Textual Inversion
    elif edit_model == 'textual_inversion':
        print(f'Evaluating for Object Addition with textual inversion .... ')
        path_images = data_path + edit_model 
        path_to_extract = os.path.join(path_images, 'object_replacement')
    
    # SDE-Edit
    elif edit_model == 'sde_edit':
        print(f'Evaluating for Object Addition with SDE-edit ......')
        path_images = data_path + edit_model 
        path_to_extract = os.path.join(path_images, 'object_replacement')
    
    # Path to extract
    print(f'############# Path to extract ###########: {path_to_extract}')
    all_dirs = os.listdir(path_to_extract)
    total_images = len(all_dirs)

    # Total paths for all the images 
    paths_all = [os.path.join(path_to_extract, a) for a in all_dirs]


    ########################## Pix2pix scales ##########################
    # Higher value will ensure faithfulness to the original image
    image_guidance_scales = [1, 1.5, 2.0, 2.5]

    # Higher guidance scale will ensure faithfulness to the text, with a drawback of reducing the fidelity of the image
    text_guidance_scale = [6.5, 7.0, 7.5, 8.0, 8.5, 9.5]

    # SDE-Edit noise scales
    sde_edit_noise_scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    ############# Define the storage for the correct edits ###########

    ##################### Pix2pix ####################
    pix2pix_total = {}
    pix2pix_correct = {}

    # Populate
    for img_guide in image_guidance_scales:
        for txt_guide in text_guidance_scale:
            key_curr = str(img_guide) + '_' + str(txt_guide)
            pix2pix_total[key_curr] = 0
            pix2pix_correct[key_curr] = 0
    

   
    ######################## SDE edit ####################
    sde_edit_total = {}
    sde_edit_correct = {} 
    
    for noise in sde_edit_noise_scales:
        sde_edit_total[str(noise)] = 0
        sde_edit_correct[str(noise)] = 0
    
    
    ################# Textual Inversion ###############
    correct_edit = 0
    total_edits = 0 

    ###########################################################

    # 
    # Iterate for every image and extract the edited images for the condition
    """ 
        Step 1: Iterate through each image -- it will have all the edits corresponding to it 
        Step 2: Inside each for loop, iterate through all the edits corresponding to the image
    """
    for path_ in paths_all:
        # Get all files 
        image_paths = os.listdir(path_)
        total_paths = [os.path.join(path_, a) for a in image_paths]
        print(len(total_paths))
        #print(total_paths)
        orig_image = [a for a in total_paths if 'unedited' in a][0]
        paths_without_orig = [a for a in total_paths if 'unedited' not in a]
        


        ################################ Obtaining the class of the given image_id ########################
        # Open
        f = open('object_new.json')
        # Load the data
        edit_data = json.load(f)
        print(f'Paths ... : {path_}')
        image_id_curr = path_.split('/')[-1]
        print(f'Current Image ID: {image_id_curr}')

        classes = list(edit_data.keys())
        class_curr = None 
        edit_instructions = []

        for cls_ in classes:
            curr_tuple = list(edit_data[cls_].keys())

            # Image_id
            if image_id_curr in curr_tuple:
                # Search the instructions from the current tuple 
                class_curr = cls_ 

        #######################################################################################################

        # Accuracy for the current edit corresponding to the image
        #correct_edit = 0
        # Total number of edits for the given image
        #total_edits = 0

        # Paths without original 
        #print(paths_without_orig)
        image_paths_without_orig = paths_without_orig

        # Total number of edits to evaluate
        print(f'Total Number of Edits to Evaluate for Object Deletion: {len(image_paths_without_orig)}')
        # TODO --- Separate out the accuracies for each image
        # Iterate through all the edits for the particular image id
        for img_ in image_paths_without_orig:
            # Pix2Pix
            if edit_model == 'pix2pix':
                # Image tag
                image_tag = img_.split('/')[-1].split('_')
                object_ = image_tag[3]
                
                # Image-tag
                img_guide = image_tag[4]
                text_guide = image_tag[5][:-4]
                key_curr = str(img_guide) + '_' + str(text_guide)
            

            # SDE_edit
            elif edit_model == 'sde_edit':
                # Image-Tag
                image_tag = img_.split('/')[-1].split('_')
                object_ = image_tag[3]
                noise_guide = image_tag[4][:-4]
                key_curr = str(noise_guide)


            # Textual Inversion
            elif edit_model == 'textual_inversion':
                image_tag = img_.split('/')[-1].split('_')
                object_ = image_tag[3]
            
            
            # Dreambooth # TODO


            # SINE # TODO 


            # Null Text # Add Tag  # TODO

            # Imagic # Add Tag  # TODO


            ######################################### Object detector part #######################################
            #image = Image.open(requests.get(url, stream=True).raw)
            image = Image.open(img_)
            texts = [["a photo of a " + str(object_), "a photo of a " + str(class_curr)]] #[["a photo of a bread", "a photo of a cup", "a photo of a finger"]]
            #print(texts)
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
            addition = 0
            # Score threshold 
            score_threshold = 0.1
            for box, score, label in zip(boxes, scores, labels):
                box = [round(i, 2) for i in box.tolist()]
                if score >= score_threshold:
                    #print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
                    if label == 0:
                        addition += 1
                    
                    if label == 1:
                        orig += 1

            
            # If original object is absent and new added object is present
            if orig == 0 and addition > 0:
                if edit_model == 'pix2pix':
                    pix2pix_correct[key_curr] += 1
                
                elif edit_model == 'sde_edit':
                    sde_edit_correct[key_curr] += 1
                
                else:
                    correct_edit += 1
            


            if edit_model == 'pix2pix':
                pix2pix_total[key_curr] += 1
            
            elif edit_model == 'sde_edit':
                sde_edit_total[key_curr] += 1
            
            else:
                total_edits += 1

            # # Total Edits
            # total_edits += 1

            break 

        

        
        break 


    # Load the edited image corresponding to the image 
    if edit_model == 'pix2pix':
        print(f'Correct: {pix2pix_correct}')
        print(f'Total: {pix2pix_total}')
    

    elif edit_model == 'sde_edit':
        print(f'Correct: {sde_edit_correct}')
        print(f'Total: {sde_edit_total}')
    
    else:
        print(f'Correct: {correct_edit}')
        print(f'Total: {total_edits}')


    return 

# Function for object replacement
def object_replacement(edit_model):
    print(f'Attribute: Object Replacement')
    # Old object has to be present
    # New object has to be present 
    # Old objects position is not changed drastically 
    
    ############################# Load Object-Detector #############################
    # Processor ffor OwlViT
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32", cache_dir='./cache')
    # OwlViT Model
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32", cache_dir = './cache')

    ############################## Put constraints based on the condition ##############################
    orig_path = data_path
    # Pix2pix
    if edit_model == 'pix2pix':
        print(f'Evaluating for Object Addition with pix2pix....')
        path_images = orig_path + edit_model
        path_to_extract = os.path.join(path_images, 'object_replacement')

    # Textual Inversion
    elif edit_model == 'textual_inversion':
        print(f'Evaluating for Object Addition with textual inversion .... ')
        path_images = orig_path + edit_model 
        path_to_extract = os.path.join(path_images, 'object_replacement')
    
    # SDE-Edit
    elif edit_model == 'sde_edit':
        print(f'Evaluating for Object Addition with SDE-edit ......')
        path_images = orig_path + edit_model 
        path_to_extract = os.path.join(path_images, 'object_replacement')
    
    # Null-Text 
    elif edit_model == 'null_text':
        print(f'Evaluating for Object Addition with Null-Text ......')
        path_images = orig_path + 'null_text_renamed' 
        path_to_extract = os.path.join(path_images, 'object_replacement')


    # imagic
    elif edit_model == 'imagic':
        print(f'Evaluating for Object Addition with SDE-edit ......')
        path_images = orig_path + edit_model 
        path_to_extract = os.path.join(path_images, 'object_replacement')

    # Path to extract
    print(f'############# Path to extract ###########: {path_to_extract}')
    all_dirs = os.listdir(path_to_extract)
    total_images = len(all_dirs)

    # Total paths for all the images 
    paths_all = [os.path.join(path_to_extract, a) for a in all_dirs]


    ############################## Create File corresponding to method and edit #####################
    result_path = os.path.join(data_path, 'results', edit_model)

    # Creata path if it doesn't exist
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        print(f'Path created: {result_path}')

    #print(f'Result Path: {result_path}')
    result_dict = {}
    ##################################################################################################



    ########################## Pix2pix scales ##########################
    # Higher value will ensure faithfulness to the original image
    image_guidance_scales = [1, 1.5, 2.0, 2.5]

    # Higher guidance scale will ensure faithfulness to the text, with a drawback of reducing the fidelity of the image
    text_guidance_scale = [6.5, 7.0, 7.5, 8.0, 8.5, 9.5]

    # SDE-Edit noise scales
    sde_edit_noise_scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    ############# Define the storage for the correct edits ###########

    ##################### Pix2pix ####################
    pix2pix_total = {}
    pix2pix_correct = {}

    # Populate
    for img_guide in image_guidance_scales:
        for txt_guide in text_guidance_scale:
            key_curr = str(img_guide) + '_' + str(txt_guide)
            pix2pix_total[key_curr] = 0
            pix2pix_correct[key_curr] = 0
    

   
    ######################## SDE edit ####################
    sde_edit_total = {}
    sde_edit_correct = {} 
    
    for noise in sde_edit_noise_scales:
        sde_edit_total[str(noise)] = 0
        sde_edit_correct[str(noise)] = 0
    
    
    ################# Textual Inversion ###############
    correct_edit = 0
    total_edits = 0 

    ###########################################################


 
    # Iterate for every image and extract the edited images for the condition
    """ 
        Step 1: Iterate through each image -- it will have all the edits corresponding to it 
        Step 2: Inside each for loop, iterate through all the edits corresponding to the image
    """
    for path_ in paths_all:
        # Get all files 
        image_paths = os.listdir(path_)
        total_paths = [os.path.join(path_, a) for a in image_paths]
        print(len(total_paths))
        #print(total_paths)
        orig_image = [a for a in total_paths if 'unedited' in a][0]
        paths_without_orig = [a for a in total_paths if 'unedited' not in a]
        


        ################################ Obtaining the class of the given image_id ########################
        # Open
        f = open('object_new.json')
        # Load the data
        edit_data = json.load(f)
        print(f'Paths ... : {path_}')
        image_id_curr = path_.split('/')[-1]
        print(f'Current Image ID: {image_id_curr}')

        classes = list(edit_data.keys())
        class_curr = None 
        edit_instructions = []

        for cls_ in classes:
            curr_tuple = list(edit_data[cls_].keys())

            # Image_id
            if image_id_curr in curr_tuple:
                # Search the instructions from the current tuple 
                class_curr = cls_ 

        #######################################################################################################

        # Accuracy for the current edit corresponding to the image
        #correct_edit = 0
        # Total number of edits for the given image
        #total_edits = 0

        # Paths without original 
        #print(paths_without_orig)
        image_paths_without_orig = paths_without_orig

        # Total number of edits to evaluate
        print(f'Total Number of Edits to Evaluate for Object Replacement: {len(image_paths_without_orig)}')
        # TODO --- Separate out the accuracies for each image
        # Iterate through all the edits for the particular image id
        for img_ in image_paths_without_orig:
            # Pix2Pix
            if edit_model == 'pix2pix':
                # Image tag
                image_tag = img_.split('/')[-1].split('_')
                object_ = image_tag[3]
                
                # Image-tag
                img_guide = image_tag[4]
                text_guide = image_tag[5][:-4]
                key_curr = str(img_guide) + '_' + str(text_guide)

            
            # SDE_edit
            elif edit_model == 'sde_edit':
                # Image-Tag
                image_tag = img_.split('/')[-1].split('_')
                object_ = image_tag[3]
                noise_guide = image_tag[4][:-4]
                key_curr = str(noise_guide)


            # Textual Inversion
            elif edit_model == 'textual_inversion':
                image_tag = img_.split('/')[-1].split('_')
                object_ = image_tag[3]
            
            # Null-Text
            elif edit_model == 'null_text':
                image_tag = img_.split('/')[-1].split('_')
                object_ = image_tag[1]
            
            # Imagic
            elif edit_model == 'imagic':
                image_tag = img_.split('/')[-1].split('_')
                object_ = image_tag[1]
            
        
            #print(f'###### Object: {object_}')
            # URL
            #url = "http://images.cocodataset.org/val2017/000000039769.jpg"

            ######################################### Object detector part #######################################
            #image = Image.open(requests.get(url, stream=True).raw)
            image = Image.open(img_)
            texts = [["a photo of a " + str(object_), "a photo of a " + str(class_curr)]] #[["a photo of a bread", "a photo of a cup", "a photo of a finger"]]
            #print(texts)
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
            addition = 0
            # Score threshold 
            score_threshold = 0.1
            for box, score, label in zip(boxes, scores, labels):
                box = [round(i, 2) for i in box.tolist()]
                if score >= score_threshold:
                    #print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
                    if label == 0:
                        addition += 1
                    
                    if label == 1:
                        orig += 1

            
            # If original object is absent and new added object is present
            if orig == 0 and addition > 0:
                if edit_model == 'pix2pix':
                    pix2pix_correct[key_curr] += 1
                    result_dict[img_] = 1
                
                elif edit_model == 'sde_edit':
                    sde_edit_correct[key_curr] += 1
                    result_dict[img_] = 1
                
                else:
                    correct_edit += 1
                    result_dict[img_] = 1
                #correct_edit += 1
            
            else:
                result_dict[img_] = 0
            
            # pix2pix
            if edit_model == 'pix2pix':
                    pix2pix_total[key_curr] += 1
            
            # sde_edit
            elif edit_model == 'sde_edit':
                sde_edit_total[key_curr] += 1
            
            # total_edits
            else:
                total_edits += 1

            # # Total Edits
            # total_edits += 1

            

    save_dict_path = os.path.join(result_path, 'object_replacement.pkl')
    with open(save_dict_path, 'wb') as f:
        pickle.dump(result_dict, f)
        
    with open(save_dict_path, 'rb') as f:
        loaded_dict = pickle.load(f)


    # Load the edited image corresponding to the image 
    if edit_model == 'pix2pix':
        print(f'Correct: {pix2pix_correct}')
        print(f'Total: {pix2pix_total}')
    

    elif edit_model == 'sde_edit':
        print(f'Correct: {sde_edit_correct}')
        print(f'Total: {sde_edit_total}')
    
    else:
        print(f'Correct: {correct_edit}')
        print(f'Total: {total_edits}')


    return 



# Function to add objects in certain position
def positional_addition(edit_model):
    print(f'Attribute: Positional Addition')
    # Old object has to be present
    # New object has to be present 
    # Old objects position is not changed drastically 
    
    ############################# Load Object-Detector #############################
    # Processor ffor OwlViT
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32", cache_dir='./cache')
    # OwlViT Model
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32", cache_dir = './cache')

    ############################## Put constraints based on the condition ##############################
    # Pix2pix
    orig_path = data_path
    if edit_model == 'pix2pix':
        print(f'Evaluating for Positional Object Addition with pix2pix....')
        path_images = orig_path + edit_model
        path_to_extract = os.path.join(path_images, 'positional_addition')

    # Textual Inversion
    elif edit_model == 'textual_inversion':
        print(f'Evaluating for Positional Object Addition with textual inversion .... ')
        path_images = orig_path + edit_model 
        path_to_extract = os.path.join(path_images, 'positional_addition')
    
    # SDE-Edit
    elif edit_model == 'sde_edit':
        print(f'Evaluating for Positional Object Addition with SDE-edit ......')
        path_images = orig_path + edit_model 
        path_to_extract = os.path.join(path_images, 'positional_addition')
    
    
    # Imagic 
    elif edit_model == 'imagic':
        print(f'Evaluating for Positional Addition with imagic')
        path_images = orig_path + edit_model 
        path_to_extract = os.path.join(path_images, 'positional_addition')
    

    # Null-Text
    elif edit_model == 'null_text':
        print(f'Evaluating for Positional Addition with Null-text')
        path_images = orig_path + 'null_text_renamed' 
        path_to_extract = os.path.join(path_images, 'positional_addition')



    # Path to extract
    print(f'############# Path to extract ###########: {path_to_extract}')
    all_dirs = os.listdir(path_to_extract)
    total_images = len(all_dirs)

    # Total paths for all the images 
    paths_all = [os.path.join(path_to_extract, a) for a in all_dirs]

    
    ############################# Results saving ################################
    result_path = os.path.join(data_path, 'results', edit_model)

    # Creata path if it doesn't exist
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        print(f'Path created: {result_path}')

    #print(f'Result Path: {result_path}')
    result_dict = {}


    ########################## Pix2pix scales ##########################
    # Higher value will ensure faithfulness to the original image
    image_guidance_scales = [1, 1.5, 2.0, 2.5]

    # Higher guidance scale will ensure faithfulness to the text, with a drawback of reducing the fidelity of the image
    text_guidance_scale = [6.5, 7.0, 7.5, 8.0, 8.5, 9.5]

    # SDE-Edit noise scales
    sde_edit_noise_scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    ############# Define the storage for the correct edits ###########

    ##################### Pix2pix ####################
    pix2pix_total = {}
    pix2pix_correct = {}

    # Populate
    for img_guide in image_guidance_scales:
        for txt_guide in text_guidance_scale:
            key_curr = str(img_guide) + '_' + str(txt_guide)
            pix2pix_total[key_curr] = 0
            pix2pix_correct[key_curr] = 0
    

   
    ######################## SDE edit ####################
    sde_edit_total = {}
    sde_edit_correct = {} 
    
    for noise in sde_edit_noise_scales:
        sde_edit_total[str(noise)] = 0
        sde_edit_correct[str(noise)] = 0
    
    
    ################# Textual Inversion ###############
    correct_edit = 0
    total_edits = 0 

    ###########################################################


    # Iterate for every image and extract the edited images for the condition
    """ 
        Step 1: Iterate through each image -- it will have all the edits corresponding to it 
        Step 2: Inside each for loop, iterate through all the edits corresponding to the image
    """
    for path_ in paths_all:
        # Get all files 
        image_paths = os.listdir(path_)
        total_paths = [os.path.join(path_, a) for a in image_paths]
        print(len(total_paths))
        #print(total_paths)
        orig_image = [a for a in total_paths if 'unedited' in a][0]
        paths_without_orig = [a for a in total_paths if 'unedited' not in a]
        

        ################################ Obtaining the class of the given image_id ########################
        # Open
        f = open('object_new.json')
        # Load the data
        edit_data = json.load(f)
        print(f'Paths ... : {path_}')
        image_id_curr = path_.split('/')[-1]
        print(f'Current Image ID: {image_id_curr}')

        classes = list(edit_data.keys())
        class_curr = None 
        edit_instructions = []

        for cls_ in classes:
            curr_tuple = list(edit_data[cls_].keys())

            # Image_id
            if image_id_curr in curr_tuple:
                # Search the instructions from the current tuple 
                class_curr = cls_ 

        #######################################################################################################

        # TODO : Need to put this outside the loop
        # Accuracy for the current edit corresponding to the image
        #correct_edit = 0
        # Total number of edits for the given image
        #total_edits = 0

        # Paths without original 
        #print(paths_without_orig)
        image_paths_without_orig = paths_without_orig

        # Total number of edits to evaluate
        print(f'Total Number of Edits to Evaluate for Positional Addition: {len(image_paths_without_orig)}')
        # TODO --- Separate out the accuracies for each image
        # Iterate through all the edits for the particular image id
        ####### Top, Right, Below, Left ######
        for img_ in image_paths_without_orig:
            # Pix2Pix
            if edit_model == 'pix2pix':
                # Image tag
                image_tag = img_.split('/')[-1].split('_')
                object_ = image_tag[3]
                
                # Image-tag
                img_guide = image_tag[4]
                text_guide = image_tag[5][:-4]
                key_curr = str(img_guide) + '_' + str(text_guide)
            
            # SDE_edit
            elif edit_model == 'sde_edit':
                # Image-Tag
                image_tag = img_.split('/')[-1].split('_')
                object_ = image_tag[3]
                noise_guide = image_tag[4][:-4]
                key_curr = str(noise_guide)


            # Textual Inversion
            elif edit_model == 'textual_inversion':
                image_tag = img_.split('/')[-1].split('_')
                object_ = image_tag[3]
            

            # Null-Text
            elif edit_model == 'null_text':
                image_tag = img_.split('/')[-1].split('_')
                object_ = image_tag[1]

            # Imagic
            elif edit_model == 'imagic':
                image_tag = img_.split('/')[-1].split('_')
                object_ = image_tag[1]
            

            # Extract the object tag and the position
            object_breakdown = object_.split(' ')
            pos = object_breakdown[-1]

            # Obtain the object text
            if pos == 'below':
                object_ = "".join(object_breakdown[:-1])
            
            # Obtain the object text
            elif pos == 'top' or pos == 'left' or pos == 'right':
                object_ = "".join(object_breakdown[:-2])

            print(f'Object ... {object_}')

            #######################  Evaluation part #################
            image = Image.open(img_)
            texts = [["a photo of a " + str(object_), "a photo of a " + str(class_curr)]] #[["a photo of a bread", "a photo of a cup", "a photo of a finger"]]
            #print(texts)
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
                        if edit_model == 'pix2pix':
                            pix2pix_correct[key_curr] += 1
                            result_dict[img_] = 1
                        
                        elif edit_model == 'sde_edit':
                            sde_edit_correct[key_curr] += 1
                            result_dict[img_] = 1
                        
                        else:
                            correct_edit += 1
                            result_dict[img_] = 1
                    
                    else:
                        result_dict[img_] = 0
                
                # Top
                elif pos == 'top':
                    if mean_y_addition > mean_y_original:
                        #correct_edit += 1
                        if edit_model == 'pix2pix':
                            pix2pix_correct[key_curr] += 1
                            result_dict[img_] = 1
                        
                        elif edit_model == 'sde_edit':
                            sde_edit_correct[key_curr] += 1
                            result_dict[img_] = 1
                        
                        else:
                            correct_edit += 1
                            result_dict[img_] = 1
                    
                    else:
                        result_dict[img_] = 0

                # Left
                elif pos == 'left':
                    if mean_x_addition < mean_x_original:
                        #correct_edit += 1
                        if edit_model == 'pix2pix':
                            pix2pix_correct[key_curr] += 1
                            result_dict[img_] = 1
                        
                        elif edit_model == 'sde_edit':
                            sde_edit_correct[key_curr] += 1
                            result_dict[img_] = 1
                        
                        else:
                            correct_edit += 1
                            result_dict[img_] = 1
                    
                    else:
                        result_dict[img_] = 0
                
                # Right
                elif pos == 'right':
                    if mean_x_addition > mean_x_original:
                        #correct_edit += 1
                        if edit_model == 'pix2pix':
                            pix2pix_correct[key_curr] += 1
                            result_dict[img_] = 1
                        
                        elif edit_model == 'sde_edit':
                            sde_edit_correct[key_curr] += 1
                            result_dict[img_] = 1
                        
                        else:
                            correct_edit += 1
                            result_dict[img_] = 1
                    
                    else:
                        result_dict[img_] = 0
                

                # Correct edit
                #correct_edit += 1

                #break 

            else:
                result_dict[img_] = 0


            
            # Total Edits
            #total_edits += 1
            if edit_model == 'pix2pix':
                pix2pix_total[key_curr] += 1
            
            elif edit_model == 'sde_edit':
                sde_edit_total[key_curr] += 1
            
            else:
                total_edits += 1

         
    # 
    save_dict_path = os.path.join(result_path, 'positional_addition.pkl')
    with open(save_dict_path, 'wb') as f:
        pickle.dump(result_dict, f)
        
    with open(save_dict_path, 'rb') as f:
        loaded_dict = pickle.load(f)

    
    
    # Load the edited image corresponding to the image 
    if edit_model == 'pix2pix':
        print(f'Correct: {pix2pix_correct}')
        print(f'Total: {pix2pix_total}')
    

    elif edit_model == 'sde_edit':
        print(f'Correct: {sde_edit_correct}')
        print(f'Total: {sde_edit_total}')
    
    else:
        print(f'Correct: {correct_edit}')
        print(f'Total: {total_edits}')

    
    return 



# Function to manipulate size
def size(edit_model):
    print(f'Evaluating for size')
    #print(f'Attribute: Positional Addition')
    # Old object has to be present
    # New object has to be present 
    # Old objects position is not changed drastically 
    
    ############################# Load Object-Detector #############################
    # Processor ffor OwlViT
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32", cache_dir='./cache')
    # OwlViT Model
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32", cache_dir = './cache')

    ############################## Put constraints based on the condition ##############################
    # Pix2pix
    if edit_model == 'pix2pix':
        print(f'Evaluating for Object Addition with pix2pix....')
        path_images = data_path + edit_model
        path_to_extract = os.path.join(path_images, 'size')

    # Textual Inversion
    elif edit_model == 'textual_inversion':
        print(f'Evaluating for Object Addition with textual inversion .... ')
        path_images = data_path + edit_model 
        path_to_extract = os.path.join(path_images, 'size')
    
    # SDE-Edit
    elif edit_model == 'sde_edit':
        print(f'Evaluating for Object Addition with SDE-edit ......')
        path_images = data_path + edit_model 
        path_to_extract = os.path.join(path_images, 'size')
    
    # Path to extract
    print(f'############# Path to extract ###########: {path_to_extract}')
    all_dirs = os.listdir(path_to_extract)
    total_images = len(all_dirs)

    # Total paths for all the images 
    paths_all = [os.path.join(path_to_extract, a) for a in all_dirs]

    ########################## Pix2pix scales ##########################
    # Higher value will ensure faithfulness to the original image
    image_guidance_scales = [1, 1.5, 2.0, 2.5]

    # Higher guidance scale will ensure faithfulness to the text, with a drawback of reducing the fidelity of the image
    text_guidance_scale = [6.5, 7.0, 7.5, 8.0, 8.5, 9.5]

    # SDE-Edit noise scales
    sde_edit_noise_scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    ############# Define the storage for the correct edits ###########

    ##################### Pix2pix ####################
    pix2pix_total = {}
    pix2pix_correct = {}

    # Populate
    for img_guide in image_guidance_scales:
        for txt_guide in text_guidance_scale:
            key_curr = str(img_guide) + '_' + str(txt_guide)
            pix2pix_total[key_curr] = 0
            pix2pix_correct[key_curr] = 0
    

   
    ######################## SDE edit ####################
    sde_edit_total = {}
    sde_edit_correct = {} 
    
    for noise in sde_edit_noise_scales:
        sde_edit_total[str(noise)] = 0
        sde_edit_correct[str(noise)] = 0
    
    
    ################# Textual Inversion ###############
    correct_edit = 0
    total_edits = 0 


    
    # Iterate for every image and extract the edited images for the condition
    """ 
        Step 1: Iterate through each image -- it will have all the edits corresponding to it 
        Step 2: Inside each for loop, iterate through all the edits corresponding to the image
    """
    for path_ in paths_all:
        # Get all files 
        image_paths = os.listdir(path_)
        total_paths = [os.path.join(path_, a) for a in image_paths]
        print(len(total_paths))
        #print(total_paths)
        orig_image = [a for a in total_paths if 'unedited' in a][0]
        paths_without_orig = [a for a in total_paths if 'unedited' not in a]
        
        
        ################################ Obtaining the class of the given image_id ########################
        # Open
        f = open('object_new.json')
        # Load the data
        edit_data = json.load(f)
        print(f'Paths ... : {path_}')
        image_id_curr = path_.split('/')[-1]
        print(f'Current Image ID: {image_id_curr}')
        
        classes = list(edit_data.keys())
        class_curr = None 
        edit_instructions = []

        for cls_ in classes:
            curr_tuple = list(edit_data[cls_].keys())

            # Image_id
            if image_id_curr in curr_tuple:
                # Search the instructions from the current tuple 
                class_curr = cls_ 

        #######################################################################################################

        # TODO : Need to put this outside the loop
        # Accuracy for the current edit corresponding to the image
        correct_edit = 0
        # Total number of edits for the given image
        total_edits = 0

        # Paths without original 
        #print(paths_without_orig)
        image_paths_without_orig = paths_without_orig

        # Total number of edits to evaluate
        print(f'Total Number of Edits to Evaluate for Positional Addition: {len(image_paths_without_orig)}')
        # TODO --- Separate out the accuracies for each image
        # Iterate through all the edits for the particular image id
        ####### Top, Right, Below, Left ######
        for img_ in image_paths_without_orig:
            # Pix2Pix
            if edit_model == 'pix2pix':
                # Image tag
                image_tag = img_.split('/')[-1].split('_')
                #object_ = image_tag[3]
                #print(image_tag)
                # Image-tag
                size = image_tag[2]
                img_guide = image_tag[3]
                text_guide = image_tag[4][:-4]
                key_curr = str(img_guide) + '_' + str(text_guide)
            

            # SDE_edit
            elif edit_model == 'sde_edit':
                # Image-Tag
                image_tag = img_.split('/')[-1].split('_')
                size = image_tag[2]
                #object_ = image_tag[3]
                #print(image_tag)
                noise_guide = image_tag[3][:-4]
                key_curr = str(noise_guide)


            # Textual Inversion
            elif edit_model == 'textual_inversion':
                image_tag = img_.split('/')[-1].split('_')
                size = image_tag[2]
                #print(image_tag)
                #object_ = image_tag[3]
            

            # Extract the object tag and the position
            #object_breakdown = object_.split(' ')
            #pos = object_breakdown[-1]

            #print(image_tag)
            #print(size)

            ########################### Evaluation part for old image ######################
            image = Image.open(orig_image)
            texts = [["a photo of a " + str(class_curr)]] #[["a photo of a bread", "a photo of a cup", "a photo of a finger"]]
            #print(texts)
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
            # Score threshold 
            score_threshold = 0.1
            box_original = None 
            #box_original = None 
            for box, score, label in zip(boxes, scores, labels):
                box = [round(i, 2) for i in box.tolist()]
                if score >= score_threshold:
                    #print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
                    if label == 0:
                        orig += 1
                        box_original = box 


            ################################################################################

            #######################  Evaluation part for new image #################
            image = Image.open(img_)
            texts = [["a photo of a " + str(class_curr)]] #[["a photo of a bread", "a photo of a cup", "a photo of a finger"]]
            #print(texts)
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
                   
            ###################################################################################
            # If the detector is working properly for the first image
            if orig > 0: 
                # Check if the addition is fine or not 
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

                    # Size
                    if size == 'large':
                        if area_addition > area_original:
                            #correct_edit += 1
                            if edit_model == 'pix2pix':
                                pix2pix_correct[key_curr] += 1
                            
                            elif edit_model == 'sde_edit':
                                sde_edit_correct[key_curr] += 1
                            
                            else:
                                correct_edit += 1

                    # 
                    elif size == 'small':
                        if area_addition < area_original:
                            #correct_edit += 1
                            if edit_model == 'pix2pix':
                                pix2pix_correct[key_curr] += 1
                            
                            elif edit_model == 'sde_edit':
                                sde_edit_correct[key_curr] += 1
                            
                            else:
                                correct_edit += 1
            


            # Total Edits
            if edit_model == 'pix2pix':
                pix2pix_total[key_curr] += 1
            
            elif edit_model == 'sde_edit':
                sde_edit_total[key_curr] += 1
            
            else:
                total_edits += 1


            #total_edits += 1

            break 


            ###################### 
        


        break 


    # Print
    # Load the edited image corresponding to the image 
    if edit_model == 'pix2pix':
        print(f'Correct: {pix2pix_correct}')
        print(f'Total: {pix2pix_total}')
    

    elif edit_model == 'sde_edit':
        print(f'Correct: {sde_edit_correct}')
        print(f'Total: {sde_edit_total}')
    
    else:
        print(f'Correct: {correct_edit}')
        print(f'Total: {total_edits}')


    return 



# Function for altering object parts
def alter_object_parts(edit_model):
    print(f'Evaluating for altering object-parts')
    #print(f'Attribute: Object Addition')
    # Old object has to be present
    # New object has to be present 
    # Old objects position is not changed drastically 
    
    ############################# Load Object-Detector #############################
    # Processor ffor OwlViT
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32", cache_dir='./cache')
    # OwlViT Model
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32", cache_dir = './cache')

    ############################## Put constraints based on the condition ##############################
    # Pix2pix
    if edit_model == 'pix2pix':
        print(f'Evaluating for Object Addition with pix2pix....')
        path_images = data_path + edit_model
        path_to_extract = os.path.join(path_images, 'alter_parts')

    # Textual Inversion
    elif edit_model == 'textual_inversion':
        print(f'Evaluating for Object Addition with textual inversion .... ')
        path_images = data_path + edit_model 
        path_to_extract = os.path.join(path_images, 'alter_parts')
    
    # SDE-Edit
    elif edit_model == 'sde_edit':
        print(f'Evaluating for Object Addition with SDE-edit ......')
        path_images = data_path + edit_model 
        path_to_extract = os.path.join(path_images, 'alter_parts')
    
    # Path to extract
    print(f'############# Path to extract ###########: {path_to_extract}')
    all_dirs = os.listdir(path_to_extract)
    total_images = len(all_dirs)

    # Total paths for all the images 
    paths_all = [os.path.join(path_to_extract, a) for a in all_dirs]


     ########################## Pix2pix scales ##########################
    # Higher value will ensure faithfulness to the original image
    image_guidance_scales = [1, 1.5, 2.0, 2.5]

    # Higher guidance scale will ensure faithfulness to the text, with a drawback of reducing the fidelity of the image
    text_guidance_scale = [6.5, 7.0, 7.5, 8.0, 8.5, 9.5]

    # SDE-Edit noise scales
    sde_edit_noise_scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    ############# Define the storage for the correct edits ###########

    ##################### Pix2pix ####################
    pix2pix_total = {}
    pix2pix_correct = {}

    # Populate
    for img_guide in image_guidance_scales:
        for txt_guide in text_guidance_scale:
            key_curr = str(img_guide) + '_' + str(txt_guide)
            pix2pix_total[key_curr] = 0
            pix2pix_correct[key_curr] = 0
    

   
    ######################## SDE edit ####################
    sde_edit_total = {}
    sde_edit_correct = {} 
    
    for noise in sde_edit_noise_scales:
        sde_edit_total[str(noise)] = 0
        sde_edit_correct[str(noise)] = 0
    
    
    ################# Textual Inversion ###############
    correct_edit = 0
    total_edits = 0 
 
    # Iterate for every image and extract the edited images for the condition
    """ 
        Step 1: Iterate through each image -- it will have all the edits corresponding to it 
        Step 2: Inside each for loop, iterate through all the edits corresponding to the image
    """
    for path_ in paths_all:
        # Get all files 
        image_paths = os.listdir(path_)
        total_paths = [os.path.join(path_, a) for a in image_paths]
        print(len(total_paths))
        #print(total_paths)
        orig_image = [a for a in total_paths if 'unedited' in a][0]
        paths_without_orig = [a for a in total_paths if 'unedited' not in a]
        


        ################################ Obtaining the class of the given image_id ########################
        # Open
        f = open('object_new.json')
        # Load the data
        edit_data = json.load(f)
        print(f'Paths ... : {path_}')
        image_id_curr = path_.split('/')[-1]
        print(f'Current Image ID: {image_id_curr}')

        classes = list(edit_data.keys())
        class_curr = None 
        edit_instructions = []

        for cls_ in classes:
            curr_tuple = list(edit_data[cls_].keys())

            # Image_id
            if image_id_curr in curr_tuple:
                # Search the instructions from the current tuple 
                class_curr = cls_ 

        #######################################################################################################

        # Accuracy for the current edit corresponding to the image
        correct_edit = 0
        # Total number of edits for the given image
        total_edits = 0

        # Paths without original 
        #print(paths_without_orig)
        image_paths_without_orig = paths_without_orig

        # Total number of edits to evaluate
        print(f'Total Number of Edits to Evaluate for Object Addition: {len(image_paths_without_orig)}')
        # TODO --- Separate out the accuracies for each image
        # Iterate through all the edits for the particular image id
        for img_ in image_paths_without_orig:
            # Pix2Pix
            if edit_model == 'pix2pix':
                # Image tag
                image_tag = img_.split('/')[-1].split('_')
                object__ = image_tag[3]
                
                # Image-tag
                img_guide = image_tag[4]
                text_guide = image_tag[5][:-4]
                key_curr = str(img_guide) + '_' + str(text_guide)
            
            # SDE_edit
            elif edit_model == 'sde_edit':
                # Image-Tag
                image_tag = img_.split('/')[-1].split('_')
                object__ = image_tag[3]
                noise_guide = image_tag[3][:-4]
                key_curr = str(noise_guide)


            # Textual Inversion
            elif edit_model == 'textual_inversion':
                image_tag = img_.split('/')[-1].split('_')
                object__ = image_tag[3]
            
            #print(object_)
            #object_ = " ".join(object__.split('')[1:])
            object_ = " ".join(object__.split(' ')[1:])
            
            
            #image = Image.open(requests.get(url, stream=True).raw)
            image = Image.open(img_)
            texts = [["a photo of a " + str(object_), "a photo of a " + str(class_curr)]] #[["a photo of a bread", "a photo of a cup", "a photo of a finger"]]
            #print(texts)
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
                #correct_edit += 1
                # Check the bounding box condition
                cx_addition, cy_addition, _, _ = box_addition
                cx_original, cy_original, w_original, h_original = box_original

                # Check the condition where the centre of the new bounding box has to be in the 
                if cx_addition < (cx_original + w_original/2) and cx_addition>(cx_original - w_original/2) and cy_addition < (cy_original + h_original/2) and cy_addition > (cy_original-h_original/2):
                    #correct_edit += 1
                    if edit_model == 'pix2pix':
                        pix2pix_correct[key_curr] += 1
                    
                    elif edit_model == 'sde_edit':
                        sde_edit_correct[key_curr] += 1
                    
                    else:
                        correct_edit += 1


            if edit_model == 'pix2pix':
                pix2pix_total[key_curr] += 1
            
            elif edit_model == 'sde_edit':
                sde_edit_total[key_curr] += 1
            
            else:
                total_edits += 1

            # Total Edits
            #total_edits += 1

            break 

        

        break 

    # Edit Model
    if edit_model == 'pix2pix':
        print(f'Correct: {pix2pix_correct}')
        print(f'Total: {pix2pix_total}')
    

    elif edit_model == 'sde_edit':
        print(f'Correct: {sde_edit_correct}')
        print(f'Total: {sde_edit_total}')
    
    else:
        print(f'Correct: {correct_edit}')
        print(f'Total: {total_edits}')
    
            
    return 

# Function to alter shape
def shape(edit_model):
    # 
    print(f'Evaluating for changing the shape')

    return 



# Evaluation script for the object detector
def main():
    # Argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls_label", default='sink', type=str, required=False, help="Class Label")
    
    #options = ['pix2pix', 'null_text', 'dreambooth', 'sde_edit', 'textual_inversion', 'imagic', 'sine', 'instabooth']
    parser.add_argument("--edit_model", default='pix2pix', type=str, required=False, help="Diffusion Model to use for editing")
    parser.add_argument("--edit_dimension", default='object_addition', type=str, required=False, help="Diffusion Model Editing Dimension to Use for Editing")
    args = parser.parse_args()

    models = ['pix2pix', 'null_text', 'sde_edit', 'textual_inversion', 'imagic']
    dimensions = ['object_replacement', 'positional_replacement', 'object_addition', 'positional_addition']

    for m in models:
        print(f'###### Model: {m} #######')
        for dim in dimensions:
            print(f'####### Dimension: {dim} ############')
            args.edit_model = m 
            args.edit_dimension = dim 
            # Dimension 1: ######### DONE (Created for 5 methods) ###########
            if args.edit_dimension == 'object_replacement':
                object_replacement(args.edit_model)
            
            # Dimension 2: ########### DONE (Created for 5 methods) #########
            elif args.edit_dimension == 'positional_replacement':
                positional_replacement(args.edit_model)
            
            # Dimension 3 : ########## DONE  (Created for 5 methods) ############
            elif args.edit_dimension == 'object_addition':
                object_addition(args.edit_model)

            # Dimension 4 : ######### DONE (Created for 5 methods) #############
            elif args.edit_dimension == 'positional_addition':
                positional_addition(args.edit_model)
            
            # Dimension 5 ########### DONE ############
            elif args.edit_dimension == 'size':
                size(args.edit_model)
            
            # Dimension 6 :############ DONE #############
            elif args.edit_dimension == 'object_deletion':
                object_deletion(args.edit_model)
            
            # Dimension 7 ############ DONE ###########
            elif args.edit_dimension == 'alter_object_parts':
                alter_object_parts(args.edit_model)

            # Dimension 8 - Requires an auxilliary shape classifier
            elif args.edit_dimension == 'shape':
                shape(args.edit_model)



    return 


# If main function:
if __name__ == "__main__":
    # Main function
    main()
