#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( dirname $( dirname $( dirname $SCRIPT_DIR ) ) )"
cd $PROJECT_DIR
PYTHONPATH=$PROJECT_DIR python evaluation/explainability_evaluation.py f3900b5bac544c59998acce6787bd379
