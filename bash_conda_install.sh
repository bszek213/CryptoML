#!/bin/bash

#DON'T USE ANY OF THIS, CONDA AND TF AND KERAS DO NOT WORK WELL
conda create --name deep python=3.8 -y

conda activate deep

conda install -c "nvidia/label/cuda-12.0.0" cuda-toolkit -y

#conda install -c conda-forge tensorflow-gpu -y
python3 -m pip install tensorflow

conda install -c anaconda keras-gpu -y

conda install -c "nvidia/label/cuda-12.0.0" cuda-toolkit -y

conda install -c anaconda pandas -y

conda install -c conda-forge scipy -y

conda install -c anaconda scikit-learn -y

conda install -c anaconda pip -y

conda install -c anaconda numpy -y

conda install -c anaconda scikit-learn -y

conda install -c conda-forge matplotlib -y

pip install yfinance 
