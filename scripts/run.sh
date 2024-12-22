#!/bin/bash

# Set script to exit immediately if a command fails
set -e

# Activate Python virtual environment if required
# Uncomment the following line if you have a virtual environment
# source ../venv/bin/activate

# Default path to the configuration file
CONFIG_FILE="./config/config.yaml"

# Check for passed arguments to override defaults
if [[ $# -gt 0 ]]; then
  CONFIG_FILE="$1"
fi

# Run the Python script with the specified or default config file
python ./run.py --config "$CONFIG_FILE"

# Deactivate virtual environment after script execution
# Uncomment if using a virtual environment
# deactivate
