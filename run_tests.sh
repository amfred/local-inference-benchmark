#!/bin/bash

# Install the specified Python libraries using pip
pip install -r requirements.txt

# Runs all tests by default
# To only run the local tests, export BENCHMARK_SERVER=false
# To only run the server tests, export BENCHMARK_OLLAMA=false

# The `ollama stop` command will unload everything from memory
ollama stop granite3-dense:8b
ollama stop granite3-dense:2b
ollama stop granite3-moe:3b

# Run the granite-from-ollama.py Python script
python3 inference-benchmark.py

# The `ollama ps` command will display what's loaded into memory
ollama ps