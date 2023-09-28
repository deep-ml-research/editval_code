LOG_DIR=$1
PROMPT=$2
IMG_SAVE_DIR=$3
v=$4
K=$5
seed=$6

echo $((1000-$K))

python SINE/stable_txt2img_guidance.py --ddim_eta 0.0 --n_iter 1 \
    --scale 10 --ddim_steps 100 \
    --sin_config SINE/configs/stable-diffusion/v1-inference.yaml \
    --config SINE/configs/stable-diffusion/v1-inference.yaml \
    --sin_ckpt $LOG_DIR"/checkpoints/last.ckpt" \
    --ckpt SINE/models/ldm/stable-diffusion-v4/sd-v1-4-full-ema.ckpt \
    --prompt "$PROMPT" \
    --cond_beta $v \
    --range_t_min $((1000-$K)) --range_t_max 1000 --single_guidance \
    --H 512 --W 512 --n_samples 1 \
    --outdir "$LOG_DIR" \
    --img_save_dir "$IMG_SAVE_DIR" \
    --seed $seed