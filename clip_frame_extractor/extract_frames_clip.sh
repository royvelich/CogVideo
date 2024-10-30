#!/bin/bash

# Setup env
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate diff3f


echo "hello from $(python --version) in $(which python)"



python extract_frames_using_clip.py --sampling_type clip_diff --sample_size 3
