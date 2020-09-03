#!/bin/bash

DIR=$( dirname $( dirname $( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )))

python $DIR/scripts/compute_relative_poses.py -s setup.json -i intrinsics.json -l landmarks.json -f filenames.json --dump_images 