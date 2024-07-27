#!/bin/bash


#activate venv
source /home/cody/Documents/codyCodes/partLabeler/dev-venv/bin/activate

#ensure env vars preserved
export PYTHONPATH=$PYTHONPATH:/home/cody/Documents/codyCodes/partLabeler/dev-venv/lib/python3.8/site-packages

#run labelImg
python3 -m labelImg

#deactivate venv
deactivate
