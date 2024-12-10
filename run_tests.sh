#!/bin/bash

# Install the specified Python libraries using pip
pip install -r requirements.txt

# Run the granite-from-ollama.py Python script
python3 inference-benchmark.py
