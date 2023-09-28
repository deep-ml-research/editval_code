"""  
Data-loader for reading the images from object.json and applying edits via different models

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
import os, shutil
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
import subprocess
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

# Accelerate
from accelerate import Accelerator
from accelerate.utils import write_basic_config
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel, DiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from torch.utils.data import Dataset
import itertools 
import PIL 
import math 


# Transformation Function
def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.4225, 0.4012, 0.3659), (0.2681, 0.2635, 0.2763)), # COCO mean, std
    ])


coco_path = './coco/train_2017'
coco_annotations_path = './coco/instances_train2017.json'


# COCO-Detection 
class CocoDetection(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.root = root 
        # Load the labels
        self.labels = []
        self.paths = []
        self.total_labels = []

        # Category ids
        cat_ids_total = self.coco.getCatIds()
        
        cat_ids = self.coco.loadCats(self.coco.getCatIds())
        category_mappings = {cat_id:[] for cat_id in cat_ids_total}

        for sup_cat in cat_ids:
            category_mappings[sup_cat['id']] = sup_cat['name']
        
 

        # Across image_ids ===> extract the labels for that particular image 
        for img_id in self.ids:
            self.paths.append(os.path.join(self.root, self.coco.loadImgs(img_id)[0]['file_name']))
            ann_ids = self.coco.getAnnIds(imgIds = img_id)

            # Comes with segmentation masks, bounding box coordinates, image_classes (Segmentation classes)
            target = self.coco.loadAnns(ann_ids)
            #print(target)
            #print(img_id)
            curr_label = [category_mappings[segment['category_id']] for segment in target]
            self.labels.append(curr_label)
            self.total_labels += curr_label 
        

        # Loading the correct labels
        print(f'Loading the labels corresponding to each dataset .. ')

        # Unique Labels
        self.unique_labels = list(set(self.total_labels))
        

    # Load_image
    def _load_image(self, id: int) -> Image.Image:
        print(id)
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    # Get_item_
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        

        return 
    
    # Search
    def search(self, class_label=None):
        # Class label
        print(f'Selected Class Labels: {class_label}')

        # Select the images which have the correspoonding class_image 
        self.relevant_ids = []

        # iterate
        c = 0
        num_images = 0
        # Iterate through the image_ids
        for img_id in self.ids:
            curr_label = self.labels[c]
                
            # Current_label
            if len(curr_label) < 2 and class_label in curr_label:
                # Save the image in the directory
                num_images += 1

                # Current image
                curr_image = self._load_image(img_id)

                # Save 
                save_path = './images/img_' + str(img_id) + '.png'

                # Save the current image
                curr_image.save(save_path)

            # Update 
            c += 1
        
        # Update the number of images 
        print(f'Number of images: {num_images}')


        return  
    
    # Function to save the images 
    def save(self, curr_ids, curr_path):
        # Iterate through the current ids
        for id_ in curr_ids:
            curr_image = self._load_image(id_)
            curr_image.save(curr_path + str(id_) + '.png')
        
        return 
    
    # Length
    def __len__(self) -> int:
        return len(self.ids)


# Function for Diffedit edit
def diffedit_edit(img_dict, edit_data, args):
    print(f'Into Diffedit')

    
    from Diffedit.diffedit import set_args, diffedit

    if not os.path.exists('./saved_models'):
        os.makedirs('./saved_models')

    if not os.path.exists('./saved_models/sd-v1-4-full-ema.ckpt'):
        import urllib.request
        print("Downloading model to ./saved_models/sd-v1-4-full-ema.ckpt")
        urllib.request.urlretrieve(
            "https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt",
            "./saved_models/sd-v1-4-full-ema.ckpt")
    

    # Load Model
    model, opt = set_args()

    #images = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images
    img_ids = list(img_dict.keys())
    
    # Local instructions
    local_instructions = []

    # Iterate through the image_ids
    for img_id in img_ids:
        # Load image which needs to be passed to the diffusion model
        img_curr = img_dict[img_id]

        # Edit instructions
        edit_instructions = []

        # Classes
        classes = list(edit_data.keys())
        class_curr = None 
        for cls_ in classes:
            curr_tuple = list(edit_data[cls_].keys())

            # Image_id
            if img_id in curr_tuple:
                # Search the instructions from the current tuple 
                instructions = edit_data[cls_][img_id]
                edit_instructions.append(instructions)
                
                class_curr = cls_ 
        

        # Current class
        instructions = edit_instructions[0]
        
        # attributes
        attributes = list(instructions.keys())
        
        # Iterate through the attributes
        for attr in attributes:
            # Get instricution list
            instruction_attr = instructions[attr]['to']
            
            # Object addition
            if attr == 'object_addition':
                for attribute_local in instruction_attr:
                    local_prompt = f"A photo of a {class_curr} and a {attribute_local}[SEP]A photo of a {class_curr}"
                    local_instructions.append((img_id, attr, local_prompt, attribute_local, class_curr))
            
            # Positional addition
            elif attr == 'positional_addition':
                for attribute_local in instruction_attr:
                    local_prompt = f"A photo of a {attribute_local} of a {class_curr}[SEP]A photo of a {class_curr}"
                    local_instructions.append((img_id, attr, local_prompt, attribute_local, class_curr))
            
            # Size 
            elif attr == 'size':
                for attribute_local in instruction_attr:
                    local_prompt = f"A photo of a {attribute_local} {class_curr}[SEP]A photo of a {class_curr}"
                    local_instructions.append((img_id, attr, local_prompt, attribute_local, class_curr))
                
            # Shape
            elif attr == 'shape':
                for attribute_local in instruction_attr:
                    local_prompt = f"A photo of a {attribute_local} {class_curr}[SEP]A photo of a {class_curr}"
                    local_instructions.append((img_id, attr, local_prompt, attribute_local, class_curr))
            
            # Alter parts
            elif attr == 'alter_parts':
                for attribute_local in instruction_attr:
                    attribute_local = attribute_local.replace('add ', '')
                    local_prompt = f"A photo of a {class_curr} with {attribute_local}[SEP]A photo of a {class_curr}"
                    local_instructions.append((img_id, attr, local_prompt, attribute_local, class_curr))

            elif attr == 'texture':
                for attribute_local in instruction_attr:
                    local_prompt = f"A photo of a {attribute_local} {class_curr}[SEP]A photo of a {class_curr}"
                    local_instructions.append((img_id, attr, local_prompt, attribute_local, class_curr))

            elif attr == 'color':
                for attribute_local in instruction_attr:
                    local_prompt = f"A photo of a {attribute_local} {class_curr}[SEP]A photo of a {class_curr}"
                    local_instructions.append((img_id, attr, local_prompt, attribute_local, class_curr))

            elif attr == 'background':
                for attribute_local in instruction_attr:
                    local_prompt = f"A photo of a {class_curr} in {attribute_local}[SEP]A photo of a {class_curr}"
                    local_instructions.append((img_id, attr, local_prompt, attribute_local, class_curr))

            elif attr == 'viewpoint':
                for attribute_local in instruction_attr:
                    local_prompt = f"A {attribute_local} view photo of a {class_curr}[SEP]A photo of a {class_curr}"
                    local_instructions.append((img_id, attr, local_prompt, attribute_local, class_curr))

            elif attr == 'object_replacement':
                for attribute_local in instruction_attr:
                    local_prompt = f"A photo of a {attribute_local}[SEP]A photo of a {class_curr}"
                    local_instructions.append((img_id, attr, local_prompt, attribute_local, class_curr))

            elif attr == 'style':
                for attribute_local in instruction_attr:
                    local_prompt = f"An art painting of a {class_curr} in {attribute_local} style[SEP]A photo of a {class_curr}"
                    local_instructions.append((img_id, attr, local_prompt, attribute_local, class_curr))

            elif attr == 'action':
                change_action_dict = {'sit': 'sitting', 'run': 'running', 'hit': 'hitting', 'jump': 'jumping',
                            'stand': 'standing', 'lay': 'laying'}
                for attribute_local in instruction_attr:
                    attribute_local = change_action_dict[attribute_local]
                    local_prompt = f"A photo of a {class_curr} {attribute_local}[SEP]A photo of a {class_curr}"
                    local_instructions.append((img_id, attr, local_prompt, attribute_local, class_curr))

            elif attr == 'position_replacement':
                for attribute_local in instruction_attr:
                    local_prompt = f"A photo of a {class_curr} on the {attribute_local} of the picture[SEP]A photo of a {class_curr}"
                    local_instructions.append((img_id, attr, local_prompt, attribute_local, class_curr))

            else:
                raise ValueError(f"Attribute {attr} is not implemented!")
            # To add -- other parts

    seeds = [17]            

    last_img_id = None

    log_dir = './logs/Diffedit'

    
    if not os.path.exists('./edited_images/Diffedit/'):
        os.makedirs('./edited_images/Diffedit/')

    # DONE: Add the guidance scale parameters; Save for a range of guidance scale parameters
    for instr in local_instructions:
        for seed in seeds:
            curr_image_id = instr[0]
            curr_attribute = instr[1]
            curr_prompt = instr[2]
            curr_attribute_local = instr[3]
            curr_class = instr[4]

            img_save_folder = os.path.join(os.getcwd(), f'edited_images/Diffedit/{str(curr_image_id)}/')
            if not os.path.exists(img_save_folder):
                os.mkdir(img_save_folder)
            img_save_dir = os.path.join(img_save_folder,
                    curr_attribute + '_' + curr_attribute_local + '_' + str(seed) + '.png')
            if os.path.exists(img_save_dir):
                # image already generated
                continue

            if curr_image_id != last_img_id:
                last_img_id = curr_image_id
                print("Current Image ID:", curr_image_id)

                # Save the unedited version
                curr_image = img_dict[curr_image_id]
                img_dir = os.path.join(img_save_folder, str(curr_image_id) + '_unedited.png')
                if not os.path.exists(img_dir):
                    curr_image.save(img_dir)

            res = diffedit(model, os.path.join(img_save_folder, str(curr_image_id) + '_unedited.png'),
                         src_prompt=curr_prompt.split('[SEP]')[1], dst_prompt=curr_prompt.split('[SEP]')[0], 
                         seed=seed, opt=opt)

            Image.fromarray(res[0]).save(img_save_dir)
                
            

    
    #print(edit_data)

    return 


# Function for SINE edit
def SINE_edit(img_dict, edit_data, args):
    print(f'Into SINE')

   
    from SINE.sine_edit import set_args, sine_edit

    if not os.path.exists('./saved_models'):
        os.makedirs('./saved_models')

    if not os.path.exists('./saved_models/sd-v1-4-full-ema.ckpt'):
        import urllib.request
        print("Downloading model to ./saved_models/sd-v1-4-full-ema.ckpt")
        urllib.request.urlretrieve(
            "https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt",
            "./saved_models/sd-v1-4-full-ema.ckpt")
    



    #images = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images
    img_ids = list(img_dict.keys())
    
    # Local instructions
    local_instructions = []

    # Iterate through the image_ids
    for img_id in img_ids:
        # Load image which needs to be passed to the diffusion model
        img_curr = img_dict[img_id]

        # Edit instructions
        edit_instructions = []

        # Classes
        classes = list(edit_data.keys())
        class_curr = None 
        for cls_ in classes:
            curr_tuple = list(edit_data[cls_].keys())

            # Image_id
            if img_id in curr_tuple:
                # Search the instructions from the current tuple 
                instructions = edit_data[cls_][img_id]
                edit_instructions.append(instructions)
                
                class_curr = cls_ 
        

        # Current class
        instructions = edit_instructions[0]
        
        # attributes
        attributes = list(instructions.keys())
        
        # Iterate through the attributes
        for attr in attributes:
            # Get instricution list
            instruction_attr = instructions[attr]['to']
            
            # Object addition
            if attr == 'object_addition':
                for attribute_local in instruction_attr:
                    local_prompt = f"A photo of a {class_curr} and a {attribute_local}[SEP]A photo of a {class_curr}"
                    local_instructions.append((img_id, attr, local_prompt, attribute_local, class_curr))
            
            # Positional addition
            elif attr == 'positional_addition':
                for attribute_local in instruction_attr:
                    local_prompt = f"A photo of a {attribute_local} of a {class_curr}[SEP]A photo of a {class_curr}"
                    local_instructions.append((img_id, attr, local_prompt, attribute_local, class_curr))
            
            # Size 
            elif attr == 'size':
                for attribute_local in instruction_attr:
                    local_prompt = f"A photo of a {attribute_local} {class_curr}[SEP]A photo of a {class_curr}"
                    local_instructions.append((img_id, attr, local_prompt, attribute_local, class_curr))
                
            # Shape
            elif attr == 'shape':
                for attribute_local in instruction_attr:
                    local_prompt = f"A photo of a {attribute_local} {class_curr}[SEP]A photo of a {class_curr}"
                    local_instructions.append((img_id, attr, local_prompt, attribute_local, class_curr))
            
            # Alter parts
            elif attr == 'alter_parts':
                for attribute_local in instruction_attr:
                    attribute_local = attribute_local.replace('add ', '')
                    local_prompt = f"A photo of a {class_curr} with {attribute_local}[SEP]A photo of a {class_curr}"
                    local_instructions.append((img_id, attr, local_prompt, attribute_local, class_curr))

            elif attr == 'texture':
                for attribute_local in instruction_attr:
                    local_prompt = f"A photo of a {attribute_local} {class_curr}[SEP]A photo of a {class_curr}"
                    local_instructions.append((img_id, attr, local_prompt, attribute_local, class_curr))

            elif attr == 'color':
                for attribute_local in instruction_attr:
                    local_prompt = f"A photo of a {attribute_local} {class_curr}[SEP]A photo of a {class_curr}"
                    local_instructions.append((img_id, attr, local_prompt, attribute_local, class_curr))

            elif attr == 'background':
                for attribute_local in instruction_attr:
                    local_prompt = f"A photo of a {class_curr} in {attribute_local}[SEP]A photo of a {class_curr}"
                    local_instructions.append((img_id, attr, local_prompt, attribute_local, class_curr))

            elif attr == 'viewpoint':
                for attribute_local in instruction_attr:
                    local_prompt = f"A {attribute_local} view photo of a {class_curr}[SEP]A photo of a {class_curr}"
                    local_instructions.append((img_id, attr, local_prompt, attribute_local, class_curr))

            elif attr == 'object_replacement':
                for attribute_local in instruction_attr:
                    local_prompt = f"A photo of a {attribute_local}[SEP]A photo of a {class_curr}"
                    local_instructions.append((img_id, attr, local_prompt, attribute_local, class_curr))

            elif attr == 'style':
                for attribute_local in instruction_attr:
                    local_prompt = f"An art painting of a {class_curr} in {attribute_local} style[SEP]A photo of a {class_curr}"
                    local_instructions.append((img_id, attr, local_prompt, attribute_local, class_curr))

            elif attr == 'action':
                change_action_dict = {'sit': 'sitting', 'run': 'running', 'hit': 'hitting', 'jump': 'jumping',
                            'stand': 'standing', 'lay': 'laying'}
                for attribute_local in instruction_attr:
                    attribute_local = change_action_dict[attribute_local]
                    local_prompt = f"A photo of a {class_curr} {attribute_local}[SEP]A photo of a {class_curr}"
                    local_instructions.append((img_id, attr, local_prompt, attribute_local, class_curr))

            elif attr == 'position_replacement':
                for attribute_local in instruction_attr:
                    local_prompt = f"A photo of a {class_curr} on the {attribute_local} of the picture[SEP]A photo of a {class_curr}"
                    local_instructions.append((img_id, attr, local_prompt, attribute_local, class_curr))

            else:
                raise ValueError(f"Attribute {attr} is not implemented!")
            # To add -- other parts
            

    # Fine-tuned model usage number of steps, increasing this results in the edit text prompt having a higher impact on output
    fine_tune_inference_steps = [400, 600] # min = 0, max = 1000

    # Increasing text_coeff results in the edit text prompt having a higher impact on output
    text_coeffs = [0.4, 0.6] # min = 0., max = 1.

    seeds = [17]

    last_img_id = None

    log_dir = './logs/SINE'

    if not os.path.exists('./edited_images/SINE/'):
        os.makedirs('./edited_images/SINE/')


    # DONE: Add the guidance scale parameters; Save for a range of guidance scale parameters
    for instr in local_instructions:
        for fine_tune_step in fine_tune_inference_steps:
            for text_coeff in text_coeffs:
                for seed in seeds:
                    curr_image_id = instr[0]
                    curr_attribute = instr[1]
                    curr_prompt = instr[2]
                    curr_attribute_local = instr[3]
                    curr_class = instr[4]


                    img_log_dir = os.path.join(os.getcwd(), log_dir.replace('./', ''), '_' + str(args.run_id))
                    model_dir = os.path.join(img_log_dir, 'checkpoints/last.ckpt')
                    img_save_folder = os.path.join(os.getcwd(), f'edited_images/SINE/{str(curr_image_id)}/')

                    img_save_dir = os.path.join(img_save_folder,
                            curr_attribute + '_' + curr_attribute_local + '_' + str(text_coeff) + '_' + str(fine_tune_step) + '_' + str(seed) + '.png')
                    if os.path.exists(img_save_dir):
                        # image already generated
                        continue

                    if curr_image_id != last_img_id:
                        last_img_id = curr_image_id

                        if not os.path.exists(img_save_folder):
                            os.mkdir(img_save_folder)
                        
                        print(f"proccessing img id: {curr_image_id}, prompt: {curr_prompt}, {text_coeff} {fine_tune_step} {seed}")

                        # Save the unedited version
                        curr_image = img_dict[curr_image_id]
                        img_dir = os.path.join(img_save_folder, str(curr_image_id) + '_unedited.png')
                        if not os.path.exists(img_dir):
                            curr_image.save(img_dir)

                        if os.path.exists(model_dir):
                            os.remove(model_dir)
                        subprocess.run(["bash", "SINE/train.sh", img_dir, str(curr_class), str(args.run_id)],
                                capture_output=False)


                        # model, sin_model, sampler, opt = set_args(img_log_dir)
                        
                    print(f"img id: {curr_image_id}, prompt: {curr_prompt}, {text_coeff} {fine_tune_step} {seed}")

                    # sine_edit(model, sin_model, sampler, opt, img_log_dir, curr_prompt, img_save_dir, float(text_coeff), int(fine_tune_step), int(seed))
                    subprocess.run(["bash", "SINE/edit.sh", img_log_dir, curr_prompt, img_save_dir, str(text_coeff), str(fine_tune_step), str(seed)],
                                capture_output=False)

            # Iterate through the local instructions
            

    
    #print(edit_data)

    return 


# Function for pix2pix edit
def pix2pix_edit(img_dict, edit_data):
    # Fill in the pix2pix
    print(f'Into pix2pix')

    # Model ID
    model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, cache_dir = './cache', safety_checker=None)
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    #images = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images
    img_ids = list(img_dict.keys())
    
    # Local instructions
    local_instructions = []

    # Iterate through the image_ids
    for img_id in img_ids:
        # Load image which needs to be passed to the diffusion model
        img_curr = img_dict[img_id]

        # Edit instructions
        edit_instructions = []

        # Classes
        classes = list(edit_data.keys())
        class_curr = None 
        for cls_ in classes:
            curr_tuple = list(edit_data[cls_].keys())

            # Image_id
            if img_id in curr_tuple:
                # Search the instructions from the current tuple 
                instructions = edit_data[cls_][img_id]
                edit_instructions.append(instructions)
                
                class_curr = cls_ 
        

        # Current class
        instructions = edit_instructions[0]
        
        # attributes
        attributes = list(instructions.keys())
        
        # Iterate through the attributes
        for attr in attributes:
            # Get instricution list
            instruction_attr = instructions[attr]['to']
            
            # Object addition
            if attr == 'object_addition':
                for attribute_local in instruction_attr:
                    local_prompt = "Add a " + attribute_local 
                    local_instructions.append((img_id, attr, local_prompt, attribute_local))
            
            # Positional addition
            elif attr == 'positional_addition':
                for attribute_local in instruction_attr:
                    local_prompt = "Add " + attribute_local + " the " + class_curr 
                    local_instructions.append((img_id, attr, local_prompt, attribute_local))
            

            # Size 
            elif attr == 'size':
                for attribute_local in instruction_attr:
                    local_prompt = "Change the size of " + class_curr + " to " + attribute_local 
                    local_instructions.append((img_id, attr, local_prompt, attribute_local))
                
            # Shape
            elif attr == 'shape':
                for attribute_local in instruction_attr:
                    local_prompt = "Change the shape of the " + class_curr + " to " + attribute_local
                    local_instructions.append((img_id, attr, local_prompt, attribute_local))
            
            # Alter parts
            elif attr == 'alter_parts':
                for attribute_local in instruction_attr:
                    local_prompt = attribute_local + " to " + class_curr 
                    local_instructions.append((img_id, attr, local_prompt, attribute_local))

            # To add -- other parts
            
            # Create the edited images
            
        # 

    # Local attributes instructions
    #local_instructions : contains tuple of the form (img_id, attribute, local_prompt (to be given to pix2pix), sub-attribute)
    # Total Number of Edits
    # Save the edited image as (img_id_attr_attribute_local.png)

    # Save-path
    save_path = './edited_images/pix2pix/'

    # Higher value will ensure faithfulness to the original image
    image_guidance_scales = [1, 1.5, 2.0, 2.5]

    # Higher guidance scale will ensure faithfulness to the text, with a drawback of reducing the fidelity of the image
    guidance_scale = [6.5, 7.0, 7.5, 8.0, 8.5, 9.5]

    # DONE: Add the guidance scale parameters; Save for a range of guidance scale parameters
    for img_guide in image_guidance_scales:
        for text_guide in guidance_scale:
            # Iterate through the local instructions
            for instr in local_instructions:
                curr_image_id = instr[0]
                curr_attribute = instr[1]
                curr_prompt = instr[2]
                curr_attribute_local = instr[3]

                # Curr image
                curr_image = img_dict[curr_image_id]
                # Save the unedited version
                curr_image.save(save_path + str(curr_image_id) + '_unedited.png')

                # Edited image
                image_edit = pipe(curr_prompt, image=curr_image, num_inference_steps=50, image_guidance_scale=img_guide, guidance_scale=text_guide).images[0]

                # Save the image
                image_edit.save(save_path + str(curr_image_id) + '_' + curr_attribute + '_' + curr_attribute_local + '_' + str(img_guide) + '_' + str(text_guide) + '.png')

                break
            
            break 
        

        break 
    
    #print(edit_data)

    return 

# Function for Null-Text inversion edit 
def null_text_edit(img_dict, edit_data):
    # print("No. of available gpus: ", \
    #     [torch.cuda.device(i) for i in range(torch.cuda.device_count())])
    # if model_id is None: 
    model_id = 'CompVis/stable-diffusion-v1-4'
    
    print(f'Into Null Text Inversion..')
    save_path = './edited_images/null_text'    
    os.makedirs(save_path, exist_ok=True)
        
    # Prepare Instructions:
    # Step 2: Prepare edit-instructions for each image in img_dict:
    img_ids = list(img_dict.keys())
    # Local instructions
    local_instructions = []
    # Iterate through the image_ids
    
    for img_id in img_ids:
        # Load image which needs to be passed to the diffusion model
        img_curr = img_dict[img_id]
        # Edit instructions
        edit_instructions = []
        # Classes
        classes = list(edit_data.keys())
        class_curr = None
        for cls_ in classes:
            curr_tuple = list(edit_data[cls_].keys())
            # Image_id
            if img_id in curr_tuple:
                # Search the instructions from the current tuple
                instructions = edit_data[cls_][img_id]
                edit_instructions.append(instructions)
                class_curr = cls_ # current object category
        
        # replace _ in class-names with space for prompts.
        class_curr = class_curr.replace('_', ' ')
        instructions = edit_instructions[0]

        # attributes
        attributes = list(instructions.keys())
        # Iterate through the attributes
        for attr in attributes:
            # Get instruction list
            instruction_attr = instructions[attr]['to']
            # Object addition
            if attr == 'object_addition':
                for attribute_local in instruction_attr:                    
                    local_prompt = "A photo of a " + class_curr +"[SEP] A photo of a " + class_curr + " and a " + attribute_local
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))

            # Positional addition
            elif attr == 'positional_addition':
                for attribute_local in instruction_attr:
                    local_prompt = "A photo of a "+class_curr+"[SEP] A photo of a "+class_curr+" and " +attribute_local
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))

            # Size
            elif attr == 'size':
                for attribute_local in instruction_attr:
                    local_prompt = "A photo of a "+class_curr+"[SEP] A photo of a "+attribute_local+" "+ class_curr
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))

            # Shape
            elif attr == 'shape':
                for attribute_local in instruction_attr:
                    local_prompt = "A photo of a " + class_curr+"[SEP] A photo of a " + class_curr + " in the shape of " + attribute_local
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))

            # Alter parts
            elif attr == 'alter_parts':
                for attribute_local in instruction_attr:
                    local_prompt = class_curr+"[SEP]"+attribute_local + " to the " + class_curr
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))

            # Texture
            elif attr == 'texture':
                for attribute_local in instruction_attr:
                    local_prompt = "A photo of a "+ class_curr+"[SEP] A photo of a "+ class_curr +" with "+ attribute_local+ " texture"
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))

            # Color
            elif attr == 'color':
                for attribute_local in instruction_attr:
                    local_prompt = "A photo of a "+ class_curr+"[SEP]"+"A photo of a "+ attribute_local + " "+ class_curr
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))            
            # Background
            elif attr == 'background':
                for attribute_local in instruction_attr:
                    local_prompt = f"A photo of a {class_curr} [SEP] A photo of a {class_curr} in {attribute_local}"
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))
            # viewpoint
            elif attr == 'viewpoint':
                for attribute_local in instruction_attr:
                    local_prompt = f"A photo of a {class_curr} [SEP] A photo of a {attribute_local} view of {class_curr}"
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))
            # Replace objects
            elif attr == 'object_replacement':
                for attribute_local in instruction_attr:
                    if len(attribute_local.split(' ')) != len(class_curr.split(' ')):
                        local_prompt = f"A photo of a {class_curr} [SEP] A photo of {attribute_local}"    
                    else:
                        local_prompt = f"A photo of a {class_curr} [SEP] A photo of a {attribute_local}"
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))
            # style
            elif attr == 'style': # multiple prompts possible in case of style-change
                for attribute_local in instruction_attr:
                    local_prompt = f"A photo of a {class_curr} [SEP] An art painting of a {class_curr} in {attribute_local} style"
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))
            # Action
            elif attr == 'action':
                change_action_dict = {'sit': 'sitting', 'run': 'running', 'hit': 'hitting', 'jump': 'jumping',
                            'stand': 'standing', 'lay': 'laying'}
                for attribute_local in instruction_attr:
                    attribute_local = change_action_dict[attribute_local]
                    local_prompt = f"A photo of a {class_curr} [SEP] A photo of a {class_curr} {attribute_local}"
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))

            elif attr == 'position_replacement':
                for attribute_local in instruction_attr:
                    local_prompt = f"A photo of a {class_curr} [SEP] A photo of a {class_curr} on the {attribute_local} of the picture"
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))

            else:
                raise ValueError(f"Attribute {attr} is not implemented!")
            
    ## local_instructions : 
    ## contains tuple of the form (img_id, attribute, class_curr, local_prompt (to be given to imagic), sub-attribute)
    ## Here local_prompt consists of "source prompt" [SEP] "target prompt"
    print("No. of instructions ", len(local_instructions))

    ## Step 2: Load Baseline LDM_STABLE
    from diffusers import DDIMScheduler
    from null_text_code.ptp_utils import AttentionStore, text2image_ldm_stable, make_controller
    from null_text_code.null_inversion import NullInversion

    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False, steps_offset=1)
    ldm_stable = StableDiffusionPipeline.from_pretrained(model_id, cache_dir=CACHE_DIR, scheduler=scheduler).to("cuda")
    try:
        ldm_stable.disable_xformers_memory_efficient_attention()
    except AttributeError:
        print("Attribute disable_xformers_memory_efficient_attention() is missing")
    tokenizer = ldm_stable.tokenizer

    print("No. of local_instructions", len(local_instructions))
    for itr, instr in tqdm(enumerate(local_instructions)):
        start_time = time.time()
        curr_image_id = instr[0]
        curr_attribute = instr[1]
        class_curr = instr[2]
        curr_prompt = instr[3]
        curr_attribute_local = instr[4]
        
        # create a subfolder with "attribute name"
        os.makedirs(save_path+'/'+curr_attribute, exist_ok=True)
        os.makedirs(save_path+'/'+curr_attribute+'/'+curr_attribute_local, exist_ok=True)
        os.makedirs(save_path+'/'+curr_attribute+'/'+curr_attribute_local+'/'+str(curr_image_id), exist_ok=True)

        # Current image
        curr_image = img_dict[curr_image_id]
        # Save the unedited version of image.
        image_path = save_path+'/'+curr_attribute+'/'+curr_attribute_local + '/'+str(curr_image_id)+'/'+ str(curr_image_id) + '_unedited.png'        
        if not os.path.exists(image_path):
            curr_image.save(image_path)

        # Edited image
        source_prompt = curr_prompt.split('[SEP]')[0].strip()
        target_prompt = curr_prompt.split('[SEP]')[1].strip()
            

        NUM_DDIM_STEPS = 50
        GUIDANCE_SCALE = 7.5
        text_guide = GUIDANCE_SCALE
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
        ## Method: Prompt To Prompt Based Image Generation via Inversion
        null_inversion = NullInversion(ldm_stable, NUM_DDIM_STEPS, GUIDANCE_SCALE, device)
        image_fname = curr_image_id.zfill(12)        
        root_path = coco_path
        img_path = root_path + image_fname+'.jpg'
        (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(img_path, source_prompt, offsets=(0,0,0,0), verbose=True)
        assert uncond_embeddings is not None, "ERROR: uncond_embeddings are NONE after null-text optimization."

        prompts = [source_prompt]
        controller = AttentionStore()

        image_inv, x_t = text2image_ldm_stable(ldm_stable, prompts, controller, latent=x_t, num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE, generator=None, uncond_embeddings=uncond_embeddings)
        
        prompts = [source_prompt, target_prompt]

        ## Tune HYPER-PARAMETERS FOR NULL TEXT inference.        
        for cross_replace_steps, self_replace_steps in [ ({'default_': .4}, 0.4), ({'default_': .4}, 0.6), ({'default_': .6}, 0.4), ({'default_': .6}, 0.6)]:

            ## Imp: Select the hyper-parameter setting according to current attribute:
            if curr_attribute == 'object_addition':
                is_replace_controller=False
                blend_word = (((class_curr,), (class_curr))) # for local edit (regional edits)        
                eq_params = {"words": (curr_attribute_local,), "values": (2,)}  # amplify attention to the words curr_attribute_local such as: "silver" by *2 
            elif curr_attribute == 'positional_addition':
                is_replace_controller=False            
                blend_word = (((class_curr,), (class_curr))) # for local edit (regional edits)        
                eq_params = {"words": (curr_attribute_local,), "values": (2,)}  # amplify attention to the words curr_attribute_local such as: "silver" by *2             
            elif curr_attribute == 'size':
                is_replace_controller=False
                blend_word = (((class_curr,), (curr_attribute_local+" "+ class_curr,))) # for local edit (regional edits)        
                eq_params = {"words": (curr_attribute_local,class_curr), "values": (2,2)}  # amplify attention to the words curr_attribute_local such as: "silver" by *2             
            elif curr_attribute == 'shape':
                is_replace_controller=False
                blend_word = (((class_curr,), (class_curr + " in the shape of " + curr_attribute_local))) # for local edit (regional edits)        
                eq_params = {"words": (curr_attribute_local,class_curr), "values": (2,2)}  # amplify attention to the words curr_attribute_local such as: "silver" by *2             
            elif curr_attribute == 'alter_parts':
                is_replace_controller=False
                blend_word = (((class_curr,), (curr_attribute_local + " to the " + class_curr))) # for local edit (regional edits)        
                eq_params = {"words": (curr_attribute_local,class_curr), "values": (2,2)}  # amplify attention to the words curr_attribute_local such as: "silver" by *2             
            elif curr_attribute == 'texture':
                is_replace_controller=False
                blend_word = (((class_curr,), (class_curr +" with "+ curr_attribute_local+ " texture"))) # for local edit (regional edits)        
                eq_params = {"words": (class_curr,curr_attribute_local), "values": (2,2)}  # amplify attention to the words curr_attribute_local such as: "silver" by *2             
            elif curr_attribute == 'color':
                is_replace_controller=False
                blend_word = (((class_curr,), (curr_attribute_local+' '+class_curr))) # for local edit (regional edits)        
                eq_params = {"words": (curr_attribute_local,class_curr), "values": (2,2)}  # amplify attention to the words curr_attribute_local such as: "silver" by *2             
            elif curr_attribute == 'background':
                is_replace_controller=False
                blend_word = (((class_curr,), (class_curr))) # for local edit (regional edits)        
                eq_params = {"words": (curr_attribute_local,), "values": (2,)}  # amplify attention to the words curr_attribute_local such as: "silver" by *2             
            elif curr_attribute == 'viewpoint':
                is_replace_controller=False
                blend_word = (((class_curr,), (curr_attribute_local+' view of '+class_curr))) # for local edit (regional edits)        
                eq_params = {"words": (curr_attribute_local,), "values": (2,)}  # amplify attention to the words curr_attribute_local such as: "silver" by *2             
            elif curr_attribute == 'object_replacement':
                is_replace_controller=True
                blend_word = (((class_curr,), (curr_attribute_local))) # for local edit (regional edits)        
                eq_params = None
            elif curr_attribute == 'style':
                is_replace_controller=False
                blend_word = (((class_curr,), (class_curr+' in '+curr_attribute_local+' style'))) # for local edit (regional edits)        
                eq_params = {"words": (curr_attribute_local,), "values": (2,)} 
            elif curr_attribute == 'action':
                is_replace_controller=False
                blend_word = (((class_curr,), (class_curr+' '+curr_attribute_local))) # for local edit (regional edits)        
                eq_params = {"words": (curr_attribute_local,), "values": (2,)}             
            elif curr_attribute == 'position_replacement':
                is_replace_controller=False
                blend_word = (((class_curr,), (class_curr+' on the '+ attribute_local))) # for local edit (regional edits)        
                eq_params = {"words": (curr_attribute_local,), "values": (2,)}             
            else:
                print("INVALID ATTRIBUTE", attr)
                break
                        
            controller = make_controller(prompts, tokenizer, is_replace_controller, cross_replace_steps, self_replace_steps, blend_word, eq_params)
            edited_images, _ = text2image_ldm_stable(ldm_stable, prompts, controller, latent=x_t, uncond_embeddings=uncond_embeddings,num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE)    
            image_edit = edited_images[1] # [0] is the default image corresponding to original one.
            # Typecast from numpy array to Image
            image_edit = Image.fromarray(image_edit)
            edit_img_save_path = save_path+'/'+curr_attribute+'/'+curr_attribute_local + '/'+str(curr_image_id)+'/'+ str(curr_image_id) +'_selfReplace' + str(self_replace_steps)+'_crossReplace' + str(cross_replace_steps['default_'])+'_text' + str(text_guide) + '.png'
            image_edit.save(edit_img_save_path)
            torch.cuda.empty_cache()
            
    
    return 



# Initializer token
# Teaching the model new concepts
#@title Setup the prompt templates for training 
imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]


#@title Setup the dataset
class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
    ):

        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]

        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example



# Create dataloader 
def create_dataloader(train_dataset, train_batch_size=1):
    return torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    

# Function for training the SD model 
def textual_inversion_signature(image, prompt, class_curr):
    # Install x-formers for better memory efficient training
    pretrained_model_name_or_path = "stabilityai/stable-diffusion-2"
    image.save('temp.png') #

    # Image corresponding to the edit
    images = [image]

    # 
    what_to_teach = "object" #@param ["object", "style"]
    #@markdown `placeholder_token` is the token you are going to use to represent your new concept (so when you prompt the model, you will say "A `` in an amusement park"). We use angle brackets to differentiate a token from other words/tokens, to avoid collision.
    placeholder_token = "S*" #@param {type:"string"}
    #@markdown `initializer_token` is a word that can summarise what your new concept is, to be used as a starting point
    initializer_token = class_curr #@param {type:"string"}
    #@markdown `placeholder_token` is the token you are going to use to represent your new concept (so when you prompt the model, you will say "A `` in an amusement park"). We use angle brackets to differentiate a token from other words/tokens, to avoid collision.

    # Templates

    #@title Load the tokenizer and add the placeholder token as a additional special token.
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        cache_dir = './cache'
    )
    
    # Original length of the tokenizer
    print(f'Original length of the tokenizer: {len(tokenizer)}')

    # Add the placeholder token in tokenizer
    num_added_tokens = tokenizer.add_tokens(placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )
     
    # Print the number of tokens added
    print(f'Number of tokens added: {num_added_tokens}')
    token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

    # Text-encoder
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", cache_dir = './cache'
    )

    # VQVAE
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", cache_dir = './cache'
    )

    # Unet
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet", cache_dir = './cache'
    )

    # Resize token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))

    print(f'Final length of the tokenizer after update: {len(tokenizer)}')

    # Get token embeddings and 
    token_embeds = text_encoder.get_input_embeddings().weight.data
    # Initialize the placeholder token-id with an initializer of the class
    token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

    ############################### Freeze params ##################################
    def freeze_params(params):
        for param in params:
            param.requires_grad = False

    # Freeze vae and unet
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())
    # Freeze all parameters except for the token embeddings in text encoder
    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )

    # freeze params except for final layer embedding;
    freeze_params(params_to_freeze)

    # Save-peth
    save_path = './edited_images/textual_inversion/'
    
    # Save-path + image name
    image.save(save_path + 'temp.png')

    # Train dataset
    train_dataset = TextualInversionDataset(
      data_root=save_path,
      tokenizer=tokenizer,
      #size=vae.sample_size,
      placeholder_token=placeholder_token,
      repeats=100,
      learnable_property=what_to_teach, #Option selected above between object and style
      center_crop=False,
      set="train",
    )

    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_config(pretrained_model_name_or_path, subfolder="scheduler", cache_dir = './cache')

    # Hyperparameters
    hyperparameters = {
        "learning_rate": 5e-04,
        "scale_lr": True,
        "max_train_steps": 2000,
        "save_steps": 250,
        "train_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": True,
        "mixed_precision": "fp16",
        "seed": 42,
        "output_dir": "sd-concept-output"
    }

    # 
    #@title Training function
    logger = get_logger(__name__)

    # Save progress
    def save_progress(text_encoder, placeholder_token_id, accelerator, save_path):
        logger.info("Saving embeddings")
        learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
        learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}
        torch.save(learned_embeds_dict, save_path)

    # Training function
    def training_function(text_encoder, vae, unet):
        train_batch_size = hyperparameters["train_batch_size"]
        gradient_accumulation_steps = hyperparameters["gradient_accumulation_steps"]
        learning_rate = hyperparameters["learning_rate"]
        max_train_steps = hyperparameters["max_train_steps"]
        output_dir = hyperparameters["output_dir"]
        gradient_checkpointing = hyperparameters["gradient_checkpointing"]

        accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=hyperparameters["mixed_precision"]
        )

        if gradient_checkpointing:
            text_encoder.gradient_checkpointing_enable()
            unet.enable_gradient_checkpointing()

        # Dataloader train
        train_dataloader = create_dataloader(train_dataset, train_batch_size)

        if hyperparameters["scale_lr"]:
            learning_rate = (
                learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
            )

        # Initialize the optimizer
        optimizer = torch.optim.AdamW(
            text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
            lr=learning_rate,
        )

        # Text-encoder
        text_encoder, optimizer, train_dataloader = accelerator.prepare(
            text_encoder, optimizer, train_dataloader
        )

        # Weight dtype
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move vae and unet to device
        vae.to(accelerator.device, dtype=weight_dtype)
        unet.to(accelerator.device, dtype=weight_dtype)

        # Keep vae in eval mode as we don't train it
        vae.eval()
        # Keep unet in train mode to enable gradient checkpointing
        unet.train()

        
        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

        # Train!
        total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        global_step = 0

        for epoch in range(num_train_epochs):
            text_encoder.train()
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(text_encoder):
                    # Convert images to latent space
                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                    latents = latents * 0.18215

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                    # Predict the noise residual
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states.to(weight_dtype)).sample

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    loss = F.mse_loss(noise_pred, target, reduction="none").mean([1, 2, 3]).mean()
                    accelerator.backward(loss)

                    # Zero out the gradients for all token embeddings except the newly added
                    # embeddings for the concept, as we only want to optimize the concept embeddings
                    if accelerator.num_processes > 1:
                        grads = text_encoder.module.get_input_embeddings().weight.grad
                    else:
                        grads = text_encoder.get_input_embeddings().weight.grad
                    # Get the index for tokens that we want to zero the grads for
                    index_grads_to_zero = torch.arange(len(tokenizer)) != placeholder_token_id
                    grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)

                    optimizer.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    if global_step % hyperparameters["save_steps"] == 0:
                        save_path = os.path.join(output_dir, f"learned_embeds-step-{global_step}.bin")
                        save_progress(text_encoder, placeholder_token_id, accelerator, save_path)

                logs = {"loss": loss.detach().item()}
                progress_bar.set_postfix(**logs)

                if global_step >= max_train_steps:
                    break

            accelerator.wait_for_everyone()


        # Create the pipeline using using the trained modules and save it.
        if accelerator.is_main_process:
            pipeline = StableDiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path,
                text_encoder=accelerator.unwrap_model(text_encoder),
                tokenizer=tokenizer,
                vae=vae,
                unet=unet,
            )
            pipeline.save_pretrained(output_dir)
            # Also save the newly trained embeddings
            save_path = os.path.join(output_dir, f"learned_embeds.bin")
            save_progress(text_encoder, placeholder_token_id, accelerator, save_path)



    ############ End of trainiing function #############
    training_function(text_encoder, vae, unet)

    # Iterate through the UNet parameters and text-encoder parameters
    for param in itertools.chain(unet.parameters(), text_encoder.parameters()):
        if param.grad is not None:
            del param.grad  # free some memory
        torch.cuda.empty_cache()


    #  Diffusers
    from diffusers import DPMSolverMultistepScheduler
    pipe = StableDiffusionPipeline.from_pretrained(
        hyperparameters["output_dir"],
        scheduler=DPMSolverMultistepScheduler.from_pretrained(hyperparameters["output_dir"], subfolder="scheduler"),
        torch_dtype=torch.float16,
    ).to("cuda")


    # 
    prompt = "a S* with a book"

    num_samples = 2 
    num_rows = 1 

    all_images = []

    for _ in range(num_rows):
        images = pipe([prompt] * num_samples, num_inference_steps=30, guidance_scale=7.5).images
        all_images.extend(images)
    
    # All images
    all_images[0].save('textinversion_temp.png')

        
    return 


# Function for Textual inversion edit 
def textual_inversion_edit(img_dict, edit_data):
    # Textual inversion edit 
    # For every image get the signature token embedding which will be used for the various edits
    print(f'Textual inversion .... ')
    save_path = './edited_images/textual_inversion/'
    
    # image-keys
    img_keys = list(img_dict.keys())
    
    # iterate through the image_ids
    for img_id in img_keys:
        # Get the current image
        curr_image = img_dict[img_id]
        
        # Use the current image to get the signature
        # Classes
        classes = list(edit_data.keys())
        class_curr = None 
        edit_instructions = []
        for cls_ in classes:
            curr_tuple = list(edit_data[cls_].keys())

            # Image_id
            if img_id in curr_tuple:
                # Search the instructions from the current tuple 
                class_curr = cls_ 
                instructions = edit_data[cls_][img_id]
                edit_instructions.append(instructions)

        
        # Current class -- optimize the S* 
        initial_prompt = "A S* " + class_curr 

    
        textual_inversion_signature(curr_image, initial_prompt, class_curr)

        # Create the edit instructions
        
        # Current class
        instructions = edit_instructions[0]
        
        # attributes
        attributes = list(instructions.keys())
        
        # Edit instructions storage
        edit_instructions = []

        # Attributes iterating
        for attr in attributes:
            # Get instricution list
            instruction_attr = instructions[attr]['to']

            # Attribute
            if attr == 'object_addition':
                for attribute_local in instruction_attr:
                    local_prompt = "A S* " + class_curr + " with " + attribute_local
                    edit_instructions.append((attr, attribute_local, local_prompt))

            # Positional addition
            elif attr == 'positional_addition':
                for attribute_local in instruction_attr:
                    local_prompt = attribute_local + " S* " + class_curr  
                    edit_instructions.append((attr, attribute_local, local_prompt))
            
            elif attr == 'size':
                for attribute_local in instruction_attr:
                    local_prompt = attribute_local + " S* " + class_curr 
                    edit_instructions.append((attr, attribute_local, local_prompt))
            
            elif attr == 'shape':
                for attribute_local in instruction_attr:
                    local_prompt = "S* " + class_curr + " in the shape of a " + attribute_local
                    edit_instructions.append((attr, attribute_local, local_prompt))

            elif attr == 'alter_parts':
                for attribute_local in instruction_attr:
                    local_prompt = attribute_local + " to " + "S* " + class_curr
                    edit_instructions.append((attr, attribute_local, local_prompt))

        



        break 


    return 


# Function for Imagic Edit
def imagic_edit(img_dict, edit_data):
    print(f'Into Imagic')
    model_id = "CompVis/stable-diffusion-v1-4"
    # Step 1: create results folder
    save_path = './edited_images/imagic/'    
    os.makedirs(save_path, exist_ok=True)

    # Step 2: Prepare edit-instructions for each image in img_dict:
    img_ids = list(img_dict.keys())
    # Local instructions
    local_instructions = []
    # Iterate through the image_ids
    
    for img_id in img_ids:
        # Load image which needs to be passed to the diffusion model
        img_curr = img_dict[img_id]
        # Edit instructions
        edit_instructions = []
        # Classes
        classes = list(edit_data.keys())
        class_curr = None
        for cls_ in classes:
            curr_tuple = list(edit_data[cls_].keys())
            # Image_id
            if img_id in curr_tuple:
                # Search the instructions from the current tuple
                instructions = edit_data[cls_][img_id]
                edit_instructions.append(instructions)
                class_curr = cls_ # current object category
        
        # # make directory for an object category:        
        class_curr = class_curr.replace('_', ' ')

        instructions = edit_instructions[0]
        # attributes
        attributes = list(instructions.keys())
        # Iterate through the attributes
        for attr in attributes:
            # Get instricution list
            instruction_attr = instructions[attr]['to']
            # Object addition
            if attr == 'object_addition':
                for attribute_local in instruction_attr:                    
                    local_prompt = "A photo of a " + class_curr + " and a " + attribute_local
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))

            # Positional addition
            elif attr == 'positional_addition':
                for attribute_local in instruction_attr:
                    local_prompt = "A photo of a "+ attribute_local
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))

            # Size
            elif attr == 'size':
                for attribute_local in instruction_attr:
                    local_prompt = "A photo of a "+attribute_local+" "+ class_curr
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))

            # Shape
            elif attr == 'shape':
                for attribute_local in instruction_attr:                    
                    local_prompt = "A photo of a " + class_curr + " in the shape of " + attribute_local
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))

            # Alter parts
            elif attr == 'alter_parts':
                for attribute_local in instruction_attr:
                    local_prompt = attribute_local + " to the " + class_curr
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))

            # Texture
            elif attr == 'texture':
                for attribute_local in instruction_attr:
                    local_prompt = "A photo of a "+ class_curr +" with "+ attribute_local+ " texture"
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))

            # Color
            elif attr == 'color':
                for attribute_local in instruction_attr:
                    local_prompt = "A photo of a "+ attribute_local + " "+ class_curr
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))            
            # Background
            elif attr == 'background':
                for attribute_local in instruction_attr:
                    local_prompt = f"A photo of a {class_curr} in {attribute_local}"
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))
            # viewpoint
            elif attr == 'viewpoint':
                for attribute_local in instruction_attr:
                    local_prompt = f"A photo of a {attribute_local} view of {class_curr}"
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))
            
            # Replace object
            elif attr == 'object_replacement':
                for attribute_local in instruction_attr:
                    local_prompt = f"A photo of a {attribute_local}"
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))
            # style
            elif attr == 'style': # multiple prompts possible in case of style-change
                for attribute_local in instruction_attr:                    
                    local_prompt = f"An art painting of a {class_curr} in {attribute_local} style"                    
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))
            # action
            elif attr == 'action':
                change_action_dict = {'sit': 'sitting', 'run': 'running', 'hit': 'hitting', 'jump': 'jumping',
                            'stand': 'standing', 'lay': 'laying'}
                for attribute_local in instruction_attr:
                    attribute_local = change_action_dict[attribute_local]
                    local_prompt = f"A photo of a {class_curr} {attribute_local}"
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))

            elif attr == 'position_replacement':
                for attribute_local in instruction_attr:
                    local_prompt = f"A photo of a {class_curr} on the {attribute_local} of the picture"
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))

            else:
                raise ValueError(f"Attribute {attr} is not implemented!")
            
    # local_instructions : contains tuple of the form (img_id, attribute, local_prompt (to be given to imagic), sub-attribute)

    # Step 3: Load the images one by one and finetune the model for each image.
    # Step 3.1: load pretrained custom_pipeline="imagic_stable_diffusion" and finetune it.
    # Step 3.2: Run inference with difference hyper-params. 
    # alpha: used for interpolation of our learned embedding with the new embedding.  alphas = [0.9, 1, 1.1, 1.2, 1.4, 1.5, 1.6, 1.8]
    alphas = [1.1, 1.2, 1.4, 1.5, 1.6]
    
    # Higher guidance scale will ensure faithfulness to the text, with the drawback of reducing the fidelity of the image
    # guidance_scale = [6.5, 7.0, 7.5, 8.0, 8.5, 9.0]
    
    guidance_scale = [7.5]
    import gc
    start_time =  time.time()

    # Iterate through the local instructions
    for itr, instr in tqdm(enumerate(local_instructions)):
        curr_image_id = instr[0]
        curr_attribute = instr[1]
        class_curr = instr[2]
        curr_prompt = instr[3]
        curr_attribute_local = instr[4]
        
        # create a subfolder with "attribute name"
        os.makedirs(save_path+'/'+curr_attribute, exist_ok=True)
        os.makedirs(save_path+'/'+curr_attribute+'/'+curr_attribute_local, exist_ok=True)
        os.makedirs(save_path+'/'+curr_attribute+'/'+curr_attribute_local+'/'+str(curr_image_id), exist_ok=True)

        # Curr image
        curr_image = img_dict[curr_image_id]
        image_path = save_path+'/'+curr_attribute+'/'+curr_attribute_local + '/'+str(curr_image_id)+'/'+ str(curr_image_id) + '_unedited.png'
        # Save the unedited version 
        if not  os.path.exists(image_path):      
            curr_image.save(image_path)

        # Load pipeline         
        from diffusers import DDIMScheduler # local imports:
        # Fix seed for generation
        generator = torch.Generator("cuda").manual_seed(0)
        pipe = DiffusionPipeline.from_pretrained(
                                model_id,
                                cache_dir=CACHE_DIR,
                                safety_checker=None,
                                use_auth_token=True,
                                local_files_only=True,
                                custom_pipeline="imagic_stable_diffusion",
                                scheduler = DDIMScheduler(\
                                            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",\
                                            clip_sample=False, set_alpha_to_one=False)
                                )
        pipe.to("cuda")
        ## to avoid OOM issue during finetuning.
        pipe.enable_xformers_memory_efficient_attention()

        # Edited image
        _ = pipe.train(curr_prompt, image=curr_image, generator=generator)

        ## Once the pipeline is trained, run inference with different alpha and text guidance scales.
        for alpha in alphas:
            for text_guide in guidance_scale:                                                                             
                image_edit = pipe(num_inference_steps=50, alpha=alpha, guidance_scale=text_guide).images[0]
                # Save the image
                edit_img_save_path = save_path+'/'+curr_attribute+'/'+curr_attribute_local + '/'+str(curr_image_id)+'/'+ str(curr_image_id) +'_alpha' + str(alpha) + '_text' + str(text_guide) + '.png'
                image_edit.save(edit_img_save_path)
                                
        del pipe, generator
        gc.collect()
        torch.cuda.empty_cache() 
        
    return 

# Instabooth edit
def instabooth_edit(img_dict, edit_data):
    return 

# Dreambooth edit
def dreambooth_edit(img_dict, edit_data):
    # Fill in the dreambooth
    print(f'Into dreambooth')
    write_basic_config()

    # Model ID
    model_name ="CompVis/stable-diffusion-v1-4"
    instance_dir="./dreambooth/in/"
    output_dir="./dreambooth/out/"
    
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    
    
    img_ids = list(img_dict.keys())

    # Local instructions
    local_instructions = []

    # Iterate through the image_ids
    for img_id in img_ids:
        # Load image which needs to be passed to the diffusion model
        img_curr = img_dict[img_id]

        # Edit instructions
        edit_instructions = []

        # Classes
        classes = list(edit_data.keys())
        class_curr = None
        for cls_ in classes:
            curr_tuple = list(edit_data[cls_].keys())

            # Image_id
            if img_id in curr_tuple:
                # Search the instructions from the current tuple
                instructions = edit_data[cls_][img_id]
                edit_instructions.append(instructions)

                class_curr = cls_

                # Current class
        instructions = edit_instructions[0]

        # attributes
        attributes = list(instructions.keys())

        # Iterate through the attributes
        for attr in attributes:
            # Get instricution list
            instruction_attr = instructions[attr]['to']

            # Object addition
            if attr == 'object_addition':
                for attribute_local in instruction_attr:
                    local_prompt = "A sdk "+ class_curr + " and a " + attribute_local
                    local_instructions.append((img_id, attr, local_prompt, attribute_local, class_curr))

            # Positional addition
            elif attr == 'positional_addition':
                for attribute_local in instruction_attr:
                    local_prompt = attribute_local + " the " + "a sdk "+ class_curr
                    local_instructions.append((img_id, attr, local_prompt, attribute_local, class_curr))


            # Size
            elif attr == 'size':
                for attribute_local in instruction_attr:
                    local_prompt = "Change the size of " + "sdk "+ class_curr + " to " + attribute_local
                    local_instructions.append((img_id, attr, local_prompt, attribute_local, class_curr))

            # Shape
            elif attr == 'shape':
                for attribute_local in instruction_attr:
                    local_prompt = "Change the shape of the " + "sdk "+ class_curr + " to " + attribute_local
                    local_instructions.append((img_id, attr, local_prompt, attribute_local, class_curr))

            # Alter parts
            elif attr == 'alter_parts':
                for attribute_local in instruction_attr:
                    local_prompt = attribute_local + " to " + "sdk "+ class_curr
                    local_instructions.append((img_id, attr, local_prompt, attribute_local, class_curr))


    # Local attributes instructions
    # local_instructions : contains tuple of the form (img_id, attribute, local_prompt (to be given to pix2pix), sub-attribute)
    # Total Number of Edits
    # Save the edited image as (img_id_attr_attribute_local.png)

    # Save-path
    save_path = './edited_images/dreambooth/'

    # Higher value will ensure faithfulness to the original image

    # Higher guidance scale will ensure faithfulness to the text, with a drawback of reducing the fidelity of the image
    guidance_scale = [6.5, 7.0, 7.5, 8.0, 8.5, 9.5]

    for img_guide in guidance_scale:
            # Iterate through the local instructions
            for instr in local_instructions:
                
                curr_image_id = instr[0]
                curr_attribute = instr[1]
                curr_prompt = instr[2]
                curr_attribute_local = instr[3]
                curr_class = instr[4]
                

                # Curr image
                curr_image = img_dict[curr_image_id]
                # Save the unedited version
                curr_image.save(save_path + str(curr_image_id) + '_unedited.png')

                # Edited image
                filename = instance_dir + "the_image.png"
                if os.path.exists(filename):
                    os.remove(filename)
                curr_image.save(instance_dir + 'the_image.png')
                
                subprocess.run(['python', './dreambooth/train_dreambooth.py', '--pretrained_model_name_or_path', model_name, '--instance_data_dir', instance_dir, '--output_dir', output_dir, '--instance_prompt', "sks" + curr_class, '--resolution', '512', '--train_batch_size', '1', '--gradient_accumulation_steps', '1', '--learning_rate', '5e-6', '--lr_scheduler', "constant", '--lr_warmup_steps', '0', '--max_train_steps', '400'])
                
                pipe = DiffusionPipeline.from_pretrained(output_dir, torch_dtype=torch.float16).to("cuda")
                image_edit = pipe(curr_prompt, num_inference_steps=50, guidance_scale=img_guide).images[0]
                

                # Save the image
                image_edit.save(
                    save_path + str(curr_image_id) + '_' + curr_attribute + '_' + curr_attribute_local + '_' + str(
                        img_guide) + '.png')

                break

            break

    return

# SDE Edit
def SDE_edit(img_dict, edit_data):
    return 

# Function to create edit 
# img_dict: {"img_id": img}
# edit_data: original dictionary storing the information from the json file
def create_edit(img_dict, edit_data, args):
    # Models: instructpix2pix, SINE, Null-text inversion, Dreambooth, Textual inversion, Imagic, Instabooth
    # Except instructpix2pix -- all other models require fine-tuning on the original image
    if args.edit_model == 'pix2pix':
        pix2pix_edit(img_dict, edit_data)
    
    elif args.edit_model == 'null_text':
        null_text_edit(img_dict, edit_data)
    
    elif args.edit_model == 'dreambooth':
        dreambooth_edit(img_dict, edit_data)
    
    elif args.edit_model == 'sde_edit':
        SDE_edit(img_dict, edit_data)
    
    elif args.edit_model == 'textual_inversion':
        textual_inversion_edit(img_dict, edit_data)
    
    elif args.edit_model == 'imagic':
        imagic_edit(img_dict, edit_data)
    
    elif args.edit_model == 'sine':
        SINE_edit(img_dict, edit_data, args)
        
    elif args.edit_model == 'diffedit':
        diffedit_edit(img_dict, edit_data, args)
    
    elif args.edit_model == 'instabooth':
        instabooth_edit(img_dict, edit_data)

    else:
        print(f'Error .... wrong choice of editing model ')

    
    return 


# Function to edit images
# cap : dataset_class
# edit_data : json containing the data
def perform_edit(cap, edit_data, args):
    # Relevant classes
    relevant_classes = list(edit_data.keys())

    # Image dictionary which stores the image-id and extracted image
    img_dict = {}
    # iterate through coco class
    for coco_class in relevant_classes:
        # Image indexes
        img_indexes_data = edit_data[coco_class]
        
        # Class images
        class_images = []
        # image indexes
        img_indexes = list(img_indexes_data.keys())
        
        # Iterate through the images
        for img_id in img_indexes:
            img_ = cap._load_image(int(img_id))
            # 
            class_images.append(img_)

            img_dict[img_id] = img_ 
        
        # Create a dictionary of {"img_id": img}
        

    # Function to create edits
    create_edit(img_dict, edit_data, args)

    return 


# Main function to create the editing loader
def main():

    # Argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls_label", default='sink', type=str, required=False, help="Class Label")
    
    #options = ['pix2pix', 'null_text', 'dreambooth', 'sde_edit', 'textual_inversion', 'imagic', 'sine', 'instabooth']
    parser.add_argument("--edit_model", default='pix2pix', type=str, required=False, help="Diffusion Model to use for editing")
    parser.add_argument("--json_dir", default='object.json', type=str, required=False, help="Path to the json file indicating image ids and their attribute shifts")
    parser.add_argument("--run_id", default='0', type=str, required=False, help="Id to save checkpoints while running multiple codes at the same time")
    args = parser.parse_args()

    
    # Open the json file
    f = open(args.json_dir)

    # Load the data
    edit_data = json.load(f)
    
    # Define the COCO dataset
    cap = CocoDetection(root = coco_path,
                            annFile = coco_annotations_path,
                            transform =_transform(224))#transforms.PILToTensor())

    # Function which performs the edit given the json references
    perform_edit(cap, edit_data, args) 

    return 


# If main function:
if __name__ == "__main__":
    # Main function
    main()


