This folder contains:

1. Javascript template for launch Amazon Mechanical Turk (AMT) Study.
2. Code to generate MTurk formatted data file (".csv") for Human Study.


To generate the input data for the human study, a folder containing the edited and unedited images with this structure must be present in this directory. The same folder must also be uploaded somewhere so it can be accessed by the mturk workers.


```
├── [method_name]
│   ├── [attribute_1]
│   |   ├── [image_id_1]
│   |   |   ├── [image_id_1]_[attribute_1]_[sub_attribute]_[hyperparams separated by '_'].png
│   |   |   ├── ...
|   |   |   └── [image_id_1]_unedited.png
│   |   ├── [image_id_2]
│   |   └── ...
│   ├── [attribute_2]
│   └── ...
```

Example:

```
├── pix2pix
│   ├── action
│   |   ├── 11635
|   |   |   ├── 11635_action_stand_1.5_6.5.png
|   |   |   └── 11635_unedited.png
│   |   └── 513188
│   └── background
```


The code below generates the CSV file for the editing method "dreambooth" with one hyper-parameter setting and 10 image samples per attribute. The `--base-link` is where the folder of images is uploaded.

```
python generate_mturk_csv.py --method dreambooth --num-hparams 1 --samples-per-attr 10 --base-link https://editbench.s3.amazonaws.com/edited_images/ --json-dir ../object.json
```
