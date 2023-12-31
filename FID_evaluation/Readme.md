### FID Evaluation : Image Fidelity

This folder contains code to evaluate FID score between a set of source images and edited images across all the editing dimensions of our benchmark EditVal.

Command to compute FID score with edited images generated by "Diffedit" on the "size" attribute/edit-dimension. First argument denotes the source images folder and the second argument denotes the edited images folder. The '.png' images should be saved directly under each folder.

```
python  fid_score.py source_images/size  pix2pix/size/target_2.0_7.5  --device cuda:0 &>> out_pix2pix_size
```

Command to compute FID score with edited images generated by "dreambooth" on the "alter_parts" attribute/edit-dimension.
```
python  fid_score.py source_images/alter_parts  dreambooth/alter_parts/target_7.5  --device cuda:0 &>> out_dreambooth_alter_parts
```

