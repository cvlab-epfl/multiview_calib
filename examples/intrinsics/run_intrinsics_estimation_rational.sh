#!/bin/bash

DIR=$( dirname $( dirname $( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )))

python $DIR/scripts/compute_intrinsics.py --folder_images ./frames -ich 6 -icw 8 -s 30 -t 24 --debug -rm