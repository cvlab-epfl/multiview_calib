#!/bin/bash

DIR=$( dirname $( dirname $( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )))

python $DIR/scripts/concatenate_relative_poses.py -s setup.json -r relative_poses.json --dump_images