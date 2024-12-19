#!/bin/bash
window_sizes=(8 10 12 13 14 15 16)
batch_sizes=( 32 64 128 200 256 )
for i in "${batch_sizes[@]}"
do
  for j in "${window_sizes[@]}"
  do
    python scripts/finetune_josh.py --config=scripts/configs/josh_finetune_config.py:"None" --name=mem_test --o_window_size="${j}" --o_batch_size="${i}" --o_steps=2 --debug=True --mode="${1}" --log_file="${1}_log.txt"
  done
done
