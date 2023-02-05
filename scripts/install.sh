#!/bin/bash

# setup a new Python environment
python3 -m venv venv
source venv/bin/activate
# now install the project requirements
pip install -r documentation/requirements.txt 