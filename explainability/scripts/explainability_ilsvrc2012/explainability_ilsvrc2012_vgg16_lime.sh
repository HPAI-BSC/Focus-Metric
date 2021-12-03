#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( dirname $( dirname $( dirname $SCRIPT_DIR ) ) )"
cd $PROJECT_DIR
PYTHONPATH=$PROJECT_DIR python explainability/explainability_mosaic.py ilsvrc2012_mosaic vgg16 vgg16_pt_imagenet.ckpt lime

