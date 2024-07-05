#!/bin/bash

git clone https://github.com/openvinotoolkit/openvino.genai.git
python -m venv ./venv
source ./venv/bin/activate
pip install -r openvino.genai/llm_bench/python/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python convert.py -m microsoft/Phi-3-mini-4k-instruct -o models/phi3-mini-4k-instruct -p FP32
python convert.py -m microsoft/Phi-3-mini-4k-instruct -o models/phi3-mini-4k-instruct -p FP16