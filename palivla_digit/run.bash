#!/bin/bash

# Check if a TPU VM name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <tpu-vm-name>"
    exit 1
fi

TPU_VM_NAME=$1
PROJECT="rail-tpus"

# Cache file for TPU name/zone mapping
CACHE_FILE="$HOME/.cache/tpus"
mkdir -p "$(dirname "$CACHE_FILE")"

# Check if the TPU info is already cached
if [ -f "$CACHE_FILE" ]; then
    CACHED_INFO=$(grep "^$TPU_VM_NAME:" "$CACHE_FILE")
    if [ -n "$CACHED_INFO" ]; then
        CACHED_ZONE=$(echo "$CACHED_INFO" | cut -d':' -f2)
        NUM_WORKERS=$(echo "$CACHED_INFO" | cut -d':' -f3)
    fi
fi

if [ -n "$CACHED_ZONE" ]; then
    ZONE=$CACHED_ZONE
else
    # Get the TPU information
    for MAYBE_ZONE in us-central1-a us-central2-b europe-west4-b; do
        TPU_INFO=$(gcloud compute tpus tpu-vm describe $TPU_VM_NAME --project=$PROJECT --zone=$MAYBE_ZONE --format=json 2>/dev/null)
        if [ $? -eq 0 ]; then
            # Cache the successful name/zone mapping and number of workers
            ZONE=$MAYBE_ZONE
            NUM_WORKERS=$(echo "$TPU_INFO" | jq '.networkEndpoints | length')
            echo "$TPU_VM_NAME:$ZONE:$NUM_WORKERS" >> "$CACHE_FILE"
            break
        fi
    done
fi

# Set the source and destination directories based on the zone
if [[ $ZONE == "europe-west4-"* ]]; then
    DEST_DIR="$TPU_VM_NAME:/nfs/nfs3/users/kstachowicz/big_vision_multimodal"
elif [[ $ZONE == "us-central2-"* ]]; then
    DEST_DIR="data-machine:/nfs/nfs2/users/kstachowicz/big_vision_multimodal"
else
    echo "Unsupported zone: $ZONE"
    exit 1
fi

echo "TPU_VM_NAME: $TPU_VM_NAME"
echo "ZONE: $ZONE"
echo "DEST_DIR: $DEST_DIR"
echo "Number of workers: $NUM_WORKERS"

# Copy the source directory to the TPU VM
rsync -avzL --exclude .git --exclude-from=.gitignore  . $DEST_DIR

# Launch the pod configuration
POD_NAME=$TPU_VM_NAME tpc launch pod_config.py

# Connect to the pod
bash ssh_pod.sh $TPU_VM_NAME

