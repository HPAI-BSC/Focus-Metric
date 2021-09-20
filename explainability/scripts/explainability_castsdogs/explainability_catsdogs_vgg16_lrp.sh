#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( dirname $( dirname $( dirname $SCRIPT_DIR ) ) )"
cd $PROJECT_DIR
PYTHONPATH=$PROJECT_DIR python explainability/explainability_mosaic.py catsdogs_mosaic vgg16 catsdogs_lrfs_pt_vgg16_e13.ckpt lrp