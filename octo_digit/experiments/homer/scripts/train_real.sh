export WANDB_MODE=disabled

python scripts/train.py \
    --config scripts/configs/octo_pretrain_config.py:vit_s \
    --name test \
