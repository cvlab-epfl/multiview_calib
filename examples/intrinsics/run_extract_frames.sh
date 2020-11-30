#!/bin/bash

DIR=$( dirname $( dirname $( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )))

mkdir frames
ffmpeg -i vid_pattern.mp4 -r 1 frames/frame_%04d.png 