#!/bin/bash
set -e

PROJECT_NAME="chope-mle"

echo "Assume miniconda installed"
echo "Enabling shell script to activate conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
echo "Creating conda environment..."
conda create -n ${PROJECT_NAME} -y python==3.8.3 pip=21.0.1
conda activate ${PROJECT_NAME}
echo "Compiling pip dependencies..."
python -m pip install pip-tools
pip-compile
echo "Install pip libraries..."
pip install -r requirements.txt

echo "Making data and models directories..."
mkdir -p data/output
mkdir -p data/preprocess
mkdir -p data/output
mkdir -p models

echo "Downloading assignment data..."
wget -P data/ https://cho.pe/dataset
unzip data/dataset -d data/
unzip data/datasets/edm_interactions.zip -d data/datasets
rm data/datasets/edm_interactions.zip
rm data/dataset
