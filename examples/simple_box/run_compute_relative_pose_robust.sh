#!/bin/bash

DIR=$( dirname $( dirname $( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )))

python $DIR/scripts/compute_relative_poses_robust.py -s setup.json -i intrinsics.json -l landmarks.json -m lmeds -n 5 -f filenames.json --dump_images