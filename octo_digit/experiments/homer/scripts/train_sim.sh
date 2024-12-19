# export WANDB_MODE=disabled

python scripts/train.py \
    --config experiments/homer/scripts/configs/mujoco_manip.py:vit_s \
    --config.dataset_kwargs.oxe_kwargs.data_dir=gs://rail-tpus-homer-v4/mujoco_rlds \
    --config.eval_interval=5000 \
    --config.viz_interval=5000 \
    --name test \
