""" 
Edit loader for SDE-EDit

Calls SDE-Edit with the image and prompt
""" 



# Libraries
import torch 
from PIL import Image
import torchvision 
import itertools 
import PIL 
import math 
import argparse
import json 
import torchvision 
#import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.datasets  import VisionDataset 
from typing import Any, Callable, List, Optional, Tuple
import os 

data_path = './edit_bench_data/'

# Transformation Function
def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.4225, 0.4012, 0.3659), (0.2681, 0.2635, 0.2763)), # COCO mean, std
    ])


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



# Function for SDE-Edit 
def sde_edit(img_dict, edit_data):
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
            from_attribute = instructions[attr]['from'][0]
            

            # Object addition: 1
            if attr == 'object_addition':
                for attribute_local in instruction_attr:
                    local_prompt = "Add a " + attribute_local 
                    local_instructions.append((img_id, attr, local_prompt, attribute_local))
            
            # Positional addition: 2
            elif attr == 'positional_addition':
                for attribute_local in instruction_attr:
                    local_prompt = "Add " + attribute_local + " the " + class_curr 
                    local_instructions.append((img_id, attr, local_prompt, attribute_local))
            

            # Size : 3
            elif attr == 'size':
                for attribute_local in instruction_attr:
                    local_prompt = "Change the size of " + class_curr + " to " + attribute_local 
                    local_instructions.append((img_id, attr, local_prompt, attribute_local))
                
            # Shape : 4
            elif attr == 'shape':
                for attribute_local in instruction_attr:
                    local_prompt = "Change the shape of the " + class_curr + " to " + attribute_local
                    local_instructions.append((img_id, attr, local_prompt, attribute_local))
            
            # Alter parts : 5
            elif attr == 'alter_parts':
                for attribute_local in instruction_attr:
                    local_prompt = attribute_local + " to " + class_curr 
                    local_instructions.append((img_id, attr, local_prompt, attribute_local))


            # Viewpoint attribute: 6
            elif attr == 'viewpoint':
                for attribute_local in instruction_attr:
                    local_prompt = "Change of the viewpoint of " + class_curr + " to " + attribute_local 
                    local_instructions.append((img_id, attr, local_prompt, attribute_local))
            
            
            # Style attribute : 7
            elif attr == 'style':
                for attribute_local in instruction_attr:
                    local_prompt = "Change the style to " + attribute_local 
                    local_instructions.append((img_id, attr, local_prompt, attribute_local))
        

            # Object replacement: 8
            elif attr == 'object_replacement':
                # 
                for attribute_local in instruction_attr:
                    local_prompt = "Replace the " + class_curr + " to " + attribute_local 
                    local_instructions.append((img_id, attr, local_prompt, attribute_local))
            
            # Action: 9
            elif attr == 'action':
                for attribute_local in instruction_attr:
                    local_prompt = "Change of the action of the " + class_curr + " from " + str(from_attribute) + " to " + attribute_local
                    local_instructions.append((img_id, attr, local_prompt, attribute_local))
            

            # Background: 10
            elif attr == 'background':
                for attribute_local in instruction_attr:
                    if from_attribute != "":
                        local_prompt = "Change of the background from " + str(from_attribute) + " to " + attribute_local
                    
                    else:
                        local_prompt = "Change the background to " + attribute_local
                    
                    local_instructions.append((img_id, attr, local_prompt, attribute_local))

            
            # COLOR: 11
            elif attr == 'color':
                # Color
                for attribute_local in instruction_attr:
                    local_prompt = "Change the color of the " + class_curr + " to " + attribute_local
                    local_instructions.append((img_id, attr, local_prompt, attribute_local))
            

            # Texture: 12
            elif attr == 'texture':
                for attribute_local in instruction_attr:
                    local_prompt = "Change the texture of the " + class_curr + " to " + attribute_local
                    local_instructions.append((img_id, attr, local_prompt, attribute_local))
            

            elif attr == 'position_replacement':
                for attribute_local in instruction_attr:
                    local_prompt = "Move " + class_curr + " to " + attribute_local
                    local_instructions.append((img_id, attr, local_prompt, attribute_local)) 

    

    ############################# Run the SDE-Edit with the local instructions ##########################
    noise_levels = [0.3, 0.5,  0.7, 0.8]
    import subprocess

    for noise_scale in noise_levels:
        # Iterate through the local instructions
        for instr in local_instructions:
            # Image ID 
            curr_image_id = instr[0]

            # Attribute which is getting changed (e.g., object_addition)
            curr_attribute = instr[1]
            # Instruction / Prompt (e.g. Add a bag)
            curr_prompt = instr[2]

            # Current Attribute Local (e.g. bag)
            curr_attribute_local = instr[3]

            # Curr image
            curr_image = img_dict[curr_image_id]

            # Save the image and call 
            curr_image.save('temp.png')

            # Save it in the path  


            # Save Path
            save_path = data_path + 'sde_edit'

            # Check if the directory exists -- if not then 
            saving_path = os.path.join(save_path, curr_attribute, str(curr_image_id))
            
            # If the saving path exists
            if not os.path.exists(saving_path):
                os.makedirs(saving_path)



            # Signature
            signature = str(curr_image_id) + '_' + curr_attribute + '_' + curr_attribute_local + '_'  + str(noise_scale) + '.png'
            print(f'Signature: {signature}')
            # Final Save Path
            final_save_path = os.path.join(saving_path, signature)
            print(f'Final save path: {final_save_path}')
            # Unedited save path
            unedited_save_path = os.path.join(saving_path, str(curr_image_id) + '_unedited.png')
            
            # Save the current image
            curr_image.save(unedited_save_path)
            #final_prompt = "" + curr_prompt + ""
            #print(final_prompt)
            # Running the sub-process
            #final_save_path = "" + final_save_path + ""
            subprocess.run(['python scripts/img2img_val.py --prompt "' + curr_prompt +'" --init-img ./temp.png --strength ' + str(noise_scale) + ' --signature "' + final_save_path + '"'], shell=True)
            
        
         
            

        
    # 
    print(f'Finished loading the sub-process .. ')
    ##################################################################################################### 
    
        
    


    return 

# Function to create edit 
# img_dict: {"img_id": img}
# edit_data: original dictionary storing the information from the json file
def create_edit(img_dict, edit_data, args):
    # Models: instructpix2pix, SINE, Null-text inversion, Dreambooth, Textual inversion, Imagic, Instabooth
    # Except instructpix2pix -- all other models require fine-tuning on the original image
    if args.edit_model == 'sde_edit':
        sde_edit(img_dict, edit_data)

    else:
        print(f'Wrong Choice ... ')
    
    return 


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


coco_path = './coco/train_2017'
coco_annotations_path = './coco/instances_train2017.json'


# Main function to create the editing loader
def main():
    # Open the json file
    f = open('./scripts/object.json')

    # Argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls_label", default='sink', type=str, required=False, help="Class Label")
    
    #options = ['pix2pix', 'null_text', 'dreambooth', 'sde_edit', 'textual_inversion', 'imagic', 'sine', 'instabooth']
    parser.add_argument("--edit_model", default='sde_edit', type=str, required=False, help="Diffusion Model to use for editing")
    args = parser.parse_args()


    # Load the data
    edit_data = json.load(f)
    
    # Define the COCO dataset
    cap = CocoDetection(root = coco_path,
                            annFile = coco_annotations_path,
                            transform =_transform(512))#transforms.PILToTensor())

    # Function which performs the edit given the json references
    perform_edit(cap, edit_data, args) 

    return 


# If main function:
if __name__ == "__main__":
    # Main function
    main()
