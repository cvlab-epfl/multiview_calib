#!/bin/bash

DIR=$( dirname $( dirname $( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )))

python $DIR/scripts/bundle_adjustment.py -s setup.json -i intrinsics.json -e poses.json -l landmarks.json -f filenames.json --dump_images -c ba_config.json 