#!/bin/bash

DIR=$( dirname $( dirname $( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )))

python $DIR/scripts/global_registration.py -s setup.json -ps output/bundle_adjustment/ba_poses.json -po output/bundle_adjustment/ba_points.json -l landmarks.json -lg landmarks_global.json -f filenames.json --dump_images