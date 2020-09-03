#!/bin/bash

DIR=$( dirname $( dirname $( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )))

python $DIR/scripts/global_registration.py -ps ba_poses.json -po ba_points.json -l landmarks_global.json -f filenames.json --dump_images