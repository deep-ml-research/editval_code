""" 
Scripts for creating the benchmark for image-editing using text-2-image models

- This scripts goes through the data-samples in MSCOCO classes corresponding to the attributes and then the images are manually chosen

# Image is saved in the following format for the benchmark

{
    "Object_category": 
        {
            "img":{
                "Attribute-1": {from: [], to: []}
                "Attribute-2": {from: [], to: []}
            }
        }

}


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

# Attibute list
attributes = ['object_addition', 'alter_objects', 'positional_object_addition', 'size', 'shape']


coco_path = './coco/train_2017'
coco_annotations_path = './coco/instances_train2017.json'

""" 
{
    "Class": 
        {
            "img_path":{
                "Attribute-1": {from: [], to: []}
                "Attribute-2": {from: [], to: []}
            }
        }

}

"""


# Datastore 
datastore = {}

############################## Dataset class for COCO-Captions ###############################
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


# Dataset class for the captioning dataset
cap = CocoDetection(root = coco_path,
                        annFile = coco_annotations_path,
                        transform =_transform(224))#transforms.PILToTensor())
    

# Object addition
object_addition = ['sink', 'person', 'bench', 'cup', 'pizza']

# Arg parser
parser = argparse.ArgumentParser()
parser.add_argument("--cls_label", default='sink', type=str, required=False, help="Class Label")
#parser.add_argument("--train_annotation_file", default=None, type=str, required=False, help="path of COCO annotation file")

args = parser.parse_args()


# Cap search
#cap.search(class_label = args.cls_label) #args.cls_label)

# Extracted IDs of the relevant images
sink = [100863, 115087, 134389, 213546, 107259, 10800]
bench = [100166, 122105, 198176, 151300, 107188, 240698]
pizza = [137938, 113199, 250614, 208250, 29982,  113513]
cup = [322369, 272694, 321852, 353427, 411881, 97264]
dog = [11635, 119320, 113168, 365519, 322945, 1025]
stop_sign = [135113, 26162, 242940, 430325, 197090, 301875]
backpack = [94926, 396542, 408248, 175227, 9679, 315338]
handbag = [529668, 208815, 10546, 209346,  365471, 497600]
person = [451987, 392601, 354888, 89103, 246839, 222016, 385588]

# overall_ids
overall_ids = [sink, bench, pizza, cup, dog, stop_sign, backpack, handbag, person]

# # object_labels
object_labels = ['sink', 'bench', 'pizza', 'cup', 'dog', 'stop_sign', 'backpack', 'handbag', 'person']

c = 0
# Iterate through the object labels
for obj_label in object_labels:
    # Current path
    curr_path = './images/' + obj_label + '/'

    # Current ids
    curr_ids = overall_ids[c]
    c += 1

    cap.save(curr_ids, curr_path)


