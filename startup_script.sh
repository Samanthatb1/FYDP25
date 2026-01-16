#!/bin/bash
cd /home/jenny/FYDP25

# Set DISPLAY variable for GUI applications
export DISPLAY=:0

source src/venv/bin/activate 
pip install -r requirements.txt
python src/main.py