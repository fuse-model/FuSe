PATHS=(
    "gs://rail-tpus-kevin-central2/logs/octo/orca-scripts/vit_s_discrete_normal_2024.02.02_18.48.25"
)

STEPS=(
    "300000"
)

CONDITIONING_MODE="l"
VIDEO_DIR="2-14-24"

TIMESTEPS="100"

TEMPERATURE="1.0"

HORIZON="2"

PRED_HORIZON="4"

EXEC_HORIZON="4"

CMD="python experiments/homer/eval_on_robot.py \
    --num_timesteps $TIMESTEPS \
    --video_save_path /mount/harddrive/homer/videos/$VIDEO_DIR \
    --trajectory_save_path /mount/harddrive/homer/trajectories/ \
    $(for i in "${!PATHS[@]}"; do echo "--checkpoint_weights_path ${PATHS[$i]} "; done) \
    $(for i in "${!PATHS[@]}"; do echo "--checkpoint_step ${STEPS[$i]} "; done) \
    --im_size 256 \
    --temperature $TEMPERATURE \
    --horizon $HORIZON \
    --pred_horizon $PRED_HORIZON \
    --exec_horizon $EXEC_HORIZON \
    --blocking \
    --modality $CONDITIONING_MODE \
    --checkpoint_cache_dir /mount/harddrive/homer/checkpoints/ \
"

echo $CMD

# $CMD --goal_eep "0.3 -0.05 0.2" --initial_eep "0.3 -0.05 0.2"
$CMD --goal_eep "0.3 0.0 0.15" --initial_eep "0.3 0.0 0.15"
