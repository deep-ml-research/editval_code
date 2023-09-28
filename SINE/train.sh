
IMG_PATH=$1
CLS_WRD=$2
NAME=$3

python SINE/main.py --no_test \
    --base SINE/configs/stable-diffusion/v1-finetune_picture.yaml \
    --actual_resume ./saved_models/sd-v1-4-full-ema.ckpt \
    -t -n $NAME --gpus 0,  --logdir ./logs/SINE \
    --data_root $IMG_PATH --reg_data_root $IMG_PATH \
    --class_word $CLS_WRD 