#!/bin/bash
# Run script for OPL Ingestion Pipeline
set -e

# Make sure current directory is added to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the script with provided arguments
python main.py "$@"
