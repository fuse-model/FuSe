#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: ssh_pod.sh <pod_name>"
    exit 1
fi

TPU_VM_NAME=$1

# Cache file for TPU name/zone mapping
CACHE_FILE="$HOME/.cache/tpus"

# Check if the TPU info is already cached
if [ -f "$CACHE_FILE" ]; then
    CACHED_INFO=$(grep "^$TPU_VM_NAME:" "$CACHE_FILE")
    if [ -n "$CACHED_INFO" ]; then
        ZONE=$(echo "$CACHED_INFO" | cut -d':' -f2)
        N_WORKERS=$(echo "$CACHED_INFO" | cut -d':' -f3)
    fi
fi

# Use default values if not found in cache
ZONE=${ZONE:-europe-west4-b}
N_WORKERS=${N_WORKERS:-16}

echo "Connecting to $TPU_VM_NAME with $N_WORKERS workers in zone $ZONE..."

tmux kill-session -t tpc_${TPU_VM_NAME} || true
tmux new -d -s tpc_${TPU_VM_NAME}
for i in $(seq 0 $(($N_WORKERS - 1))); do
    TMUX_HEIGHT=$(tmux display-message -p '#{window_height}')
    TMUX_WIDTH=$(tmux display-message -p '#{window_width}')

    tmux new-window -t tpc_${TPU_VM_NAME}:$i -k
    INNER_TMUX_COMMAND="tmux a -t tpc_${TPU_VM_NAME}"
    tmux send-keys -t tpc_${TPU_VM_NAME} "gcloud compute tpus tpu-vm ssh --zone $ZONE $TPU_VM_NAME --worker=$i -- -t $INNER_TMUX_COMMAND" Enter
done
tmux a -t tpc_${TPU_VM_NAME} || tmux switch -t tpc_${TPU_VM_NAME}
