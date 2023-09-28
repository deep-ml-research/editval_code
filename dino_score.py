""" 
DINO Score extraction

"""


""" 
Get results for the methods
"""

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
from transformers import ViTImageProcessor, ViTModel

data_path = './edit_bench_data/'

# DINO Score
def compute_dino_score(m, edit_dimension):
    #  Original Path
    orig_path = data_path
    path_images = orig_path + m 
    path_to_extract = os.path.join(path_images, edit_dimension)

    # 
    print(f'############# Path to extract ###########: {path_to_extract}')
    all_dirs = os.listdir(path_to_extract)
    total_images = len(all_dirs)

    # Total paths for all the images 
    paths_all = [os.path.join(path_to_extract, a) for a in all_dirs]

    cos_add = 0
    cos_total = 0 

    for path_ in paths_all:
        # Get all files 
        image_paths = os.listdir(path_)
        total_paths = [os.path.join(path_, a) for a in image_paths]
        
        # Dreambooth
        if m == 'dreambooth':
            print(f'# Dreambooth #')
            temp = total_paths[0]
            first_string = "/".join(temp.split('/')[:-1])
            #print(first_string)
            #print(first_string)
            id_ = temp.split('/')[-2]
            orig_image_temp = first_string + "/" + id_ + "_unedited.png"
            orig_image = orig_image_temp.replace('dreambooth', 'pix2pix')
        
            #orig_image = [a for a in total_paths_temp if 'unedited' in a][0]
            #print(total_paths_temp)
            #orig_image = [a.replace('dreambooth', 'pix2pix') for a in total_paths if 'unedited' in a][0]
        
        # Any-other method
        else:
            orig_image = [a for a in total_paths if 'unedited' in a][0]
        

        # Paths without original
        paths_without_orig = [a for a in total_paths if 'unedited' not in a]

        image = Image.open(orig_image)

        processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
        model = ViTModel.from_pretrained('facebook/dino-vitb16', cache_dir = './cache').to('cuda')

        inputs = processor(images=image, return_tensors="pt").to('cuda')
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 
        from numpy import dot
        from numpy.linalg import norm


        last_hidden_states = outputs.last_hidden_state
        cls_embedding = last_hidden_states[0][0].reshape(1,-1)
        normalized_embedding_unedited = torch.nn.functional.normalize(cls_embedding).detach().cpu().numpy()
        
        # Paths
        for path_unedited in paths_without_orig:
            #if '1.6' in path_unedited:
            #print(path_unedited)
            image = Image.open(path_unedited)
            inputs = processor(images=image, return_tensors="pt").to('cuda')
            with torch.no_grad():
                outputs = model(**inputs)
        
            # 
            last_hidden_states = outputs.last_hidden_state
            #print(last_hidden_states.shape)
            cls_embedding = last_hidden_states[0][0].reshape(1,-1)
            normalized_embedding_curr = torch.nn.functional.normalize(cls_embedding).detach().cpu().numpy()

            cos_sim = dot(normalized_embedding_unedited, normalized_embedding_curr.T)/(norm(normalized_embedding_unedited)*norm(normalized_embedding_curr))
            
            cos_add += cos_sim 
            cos_total += 1
        

    

    avg_score = cos_add/cos_total
    #print(f'Avg score: {avg_score}')

    return avg_score


# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls_label", default='sink', type=str, required=False, help="Class Label")
    
    #options = ['pix2pix', 'null_text', 'dreambooth', 'sde_edit', 'textual_inversion', 'imagic', 'sine', 'instabooth']
    parser.add_argument("--edit_model", default='pix2pix', type=str, required=False, help="Diffusion Model to use for editing")
    parser.add_argument("--edit_dimension", default='object_addition', type=str, required=False, help="Diffusion Model Editing Dimension to Use for Editing")

    # arg-parser
    args = parser.parse_args()

    # Models
    models = ['Imagic', 'Pix2pix', 'SINE', 'Sde_edit', 'Diffedit', 'Textual-Inversion', 'Null-Text', 'Dreambooth']
    edit_dimensions = ['object_addition', 'object_replacement', 'position_replacement', 'positional_addition', 'size', 'alter_parts', 'style', 'color', 'background', 'action', 'viewpoint', 'texture', 'shape']
    # 

    matrix = np.zeros((len(models), len(edit_dimensions))) 
    for m in models:
        for e in edit_dimensions:
            avg_score = compute_dino_score(m, e)

            matrix[models.index(m), edit_dimensions.index(e)] = avg_score
            print(f'Avg score for Model {m} with edit-dimension {e} is : {avg_score}')
            
        
    
    # print(matrix)
    # # Save
    np.save('score.npy', matrix)






    return 

# If main function:
if __name__ == "__main__":
    # Main function
    main()
    
