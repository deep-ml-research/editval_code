# EditVal: Benchmarking Text-Guided Image Editing Models

#### Overview ####
This repository provides a dataset of images with a set of edits for each image. To evaluate an image editing method, the method must apply these edits to the images, and store the result images. These output images can be used to fairly evaluate and compare methods to each other in different types of edits (e.g., object addition, changing color, changing background). 

The dataset used for benchmarking the image editing methods is provided in `object.json` file. The file includes `image_id`s from MS-COCO, with each image having a set of target attributes and sub-attributes for performing the edits (e.g., `image_id=100166` with `object_addition` attribute and `bag` sub-attributes means that the image editing method is supposed to add a bag to the image). New data can be easily added to this file, by adding new images and their corresponding attributes. The attribute types currently supported are `[action, alter_parts, background, color, object_addition, object_replacement, position_replacement, positional_addition, shape, size, style, texture, viewpoint]`.

The code for performing the edits using several image editing methods has been provided in `edit_loader.py`. To run the methods and store the results, you can simply use the following command (currently supported methods in the code are `pix2pix`, `sine`, `diffedit`, `textual_inversion`, `dreambooth`, `imagic` and `null_text`):

```
python edit_loader.py --edit_model pix2pix
```

The output images can be used for running human studies or automatic evaluations to measure the performance of image editing methods on different types of edits. To run the automatic evaluation, the code from `evaluate_edits.py` can be used.

#### Storing Edited Images ####
For a new method, we recommend using `object.json` to generate the edited images and store them in the following directory format : ``` /edited/<edit_dimension>/<image_id>/<to_attribute>.png```. For e.g., for object-addition, one can store as :  ``` /edited/object_addition/153243/cup.png```
This will be useful to be used along with our automated evaluation procedure. In particular one can then directly use the path as an argument along with the automated evaluation to get a score if the edit is correct or not.

#### Automatic Evaluation ####
To perform auto-evaluation on an edited image from EditVal, one can use the following command:

```
python auto_eval.py --class_ <insert_class> --edit_dimension <insert_edit_dimension> --from <insert_from_editval> --to <insert_from_editval> --orig_path <original_path_of_image> --edited_path <edited_path_of_image>
```

Example: Given an edited image, a score can be obtained in the following way:

```
python auto_eval.py --class_ bench --edit_dimension object_addition --from None --to cup --orig_path ./img.png --edited_path ./edited/object_addition/153243/cup.png
```
