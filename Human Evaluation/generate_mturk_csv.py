import os
import csv
import json
import random
import copy
import argparse
import urllib.parse

def get_prompt(attr, sub_attr, class_name):
    if attr == 'object_addition':
        return f"Add {sub_attr} to the photo"
    
    elif attr == 'positional_addition':
        return f"Add {sub_attr} of the {class_name}"
    
    elif attr == 'size':
        return f"Change the size of {class_name} to {sub_attr}"
        
    elif attr == 'shape':
        return f"Change the shape of the {class_name} to {sub_attr}"
    
    elif attr == 'alter_parts':
        return f"{sub_attr} to the {class_name}"

    elif attr == 'texture':
        return f"Change the texture of the {class_name} to {sub_attr}"

    elif attr == 'color':
        return f"Change the color of the {class_name} to {sub_attr}"

    elif attr == 'background':
        return f"Change the background to {sub_attr}"

    elif attr == 'viewpoint':
        return f"Change the direction in which the photo is taken of the {class_name} to {sub_attr}"

    elif attr == 'object_replacement':
        return f"Replace the {class_name} with {sub_attr}"

    elif attr == 'style':
        return f"Change the style of the photo to {sub_attr}"

    elif attr == 'action':
        return f"Change the action of the {class_name} to {sub_attr}"

    elif attr == 'position_replacement':
        return f"Change the position of the {class_name} to {sub_attr} of the photo"

    else:
        raise ValueError(f"Attribute {attr} is not implemented!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-dir', type=str, default='../object.json',
        help="path to json file containing attributes and subattributes for images")
    parser.add_argument('--method', type=str, required=True,
        help='method name; the images shouldve been saved in a folder with this name in current directory')
    parser.add_argument('--num-hparams', type=int, required=True,
        help='number of method hyper parameters; images are saved with their name in this format: {image_id}_{attribute}_{sub_attribute}_hp1_hp2_..._.png')
    parser.add_argument('--samples-per-attr', type=int, default=10,
        help='number of samples per attribute; each sample is a pair of image_id and subattribute; for each sample all existing hyperparameters are present in the output')
    parser.add_argument('--base-link', type=str, required=True,
        help='the images must be uploaded on somewhere so their links can be used in the mturk study; the base link is the link that the folder containing the edited and original images is uploaded at.')
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()


    with open(args.json_dir, 'r') as json_file:
        edit_data = json.load(json_file)

    csv_arr = []
    data_dict = {}

    ####################################################################################
    #                                                                   
    #   images must be saved in a directory in this format:
    #   
    #   > {method_name}
    #       > {attribute_name}
    #           > {image_id}
    #               > {image_id}_{attribute}_{sub_attribute}_hp1_hp2_..._.png
    #
    ####################################################################################

    samples_per_attr = args.samples_per_attr        
    methods = [args.method]                                
    num_hparams = [args.num_hparams]                       
    seed = args.seed
        
    for m_i, method_name in enumerate(methods):
        if not os.path.isdir(method_name):
            continue
        data_dict[method_name] = {}
        
        for attr in os.listdir(method_name):
            for img_id in os.listdir(os.path.join(method_name, attr)):
                if img_id == '354888':
                    continue
                img_folder = os.path.join(method_name, attr, img_id)
                org_img_dir = os.path.join(img_folder, f'{img_id}_unedited.png')
                for file_name in os.listdir(img_folder):
                    edited_img_dir = os.path.join(img_folder, file_name)

                    if 'unedited' in file_name:
                        continue

                    sub_attr = file_name.split('_')[-(num_hparams[m_i] + 1)]
                    attr = '_'.join(file_name.split('_')[1:-(num_hparams[m_i] + 1)])

                    hparams = file_name.split('_')[-(num_hparams[m_i]):]

                    #### You can add certain conditions on the Hyper parameters here

                    for cls_ in edit_data.keys():
                        curr_tuple = list(edit_data[cls_].keys())
                        if img_id in curr_tuple:
                            class_name = cls_ 
                            break
            
                    prompt = get_prompt(attr, sub_attr, class_name)
                        
                    if attr not in data_dict[method_name]:
                        data_dict[method_name][attr] = {}
                    if (sub_attr, img_id) not in data_dict[method_name][attr]:
                        data_dict[method_name][attr][(sub_attr, img_id)] = []
                    data_dict[method_name][attr][(sub_attr, img_id)].append([org_img_dir, edited_img_dir, prompt, class_name])

    random.seed(seed)

    keys_arr = {}

    for attr in data_dict[methods[0]].keys():
        arr = list(data_dict[methods[0]][attr].keys())
        
        random.shuffle(arr)
        keys_arr[attr] = arr[:samples_per_attr]

    with open('sampled_instances.json', 'w') as f:      # can be loaded later to get the exact set of samples 
        f.write(json.dumps(keys_arr))



    for method_name in data_dict.keys():
        total_num = 0
        for attr in data_dict[method_name].keys():
            arr = list(data_dict[method_name][attr].keys())

            for key in arr:
                if not tuple(key) in keys_arr[attr]:
                    data_dict[method_name][attr].pop(key)
                    # print(method_name, attr, key)
            total_num += len(list(data_dict[method_name][attr].keys()))
        print("Method name: {} |\t\t Number of sampled (attr,subattr,img_id) tuples: {}".format(method_name, total_num))
    print() 




    write_arr = []
    for method_name in data_dict.keys():
        for attr in data_dict[method_name].keys():
            for key, val in data_dict[method_name][attr].items():
                for tmp in val:
                    tmp[0] = urllib.parse.urljoin(args.base_link, tmp[0].replace('\\', '/'))
                    tmp[1] = urllib.parse.urljoin(args.base_link, tmp[1].replace('\\', '/'))
                    write_arr.append(tmp)

    random.shuffle(write_arr)

    BATCH_SIZE = 500

    for i in range(int(len(write_arr)/BATCH_SIZE) + 1):
        with open(f'edited_images_{i}.csv', 'w', newline='') as file:
            writer = csv.writer(file)

            writer.writerow(['url_org', 'url_edit', 'prompt', 'class_name'])

            for line in write_arr[i * BATCH_SIZE: min((i + 1) * BATCH_SIZE, len(write_arr))]:
                writer.writerow(line)

