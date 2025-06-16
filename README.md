Automated Sleep Scoring - Axonators 

# Project Overview

This project focuses on automated sleep stage classification in mice, followed by colored noise correspondence. 

# Collaboratos
- Leandro Miguel Pereira Ribeiro, up202105358
- Leonor Gonçalves Franco, up202005374

# Installation
1. Clone the repository or download the code.
	- git clone https://github.com/J-u-s-tLee/Projeto-NEURO
2. Requirements:
	- Python 3.8 or higher
	- Libraries: numpy pandas scipy matplotlib seaborn scikit-learn soundfile sounddevice joblib umap-learn h5py

# Data setup
1. Download and extract both the dataset and noise samples from: https://drive.google.com/drive/folders/1IaX8c8cy3NmsI-T9Yir4y3o-28fnS61T?usp=sharing
	- The 23 .mat files should be saved inside a folder named "Continuous".
	- The 3 .wav files should be saved inside a folder named "sounds".
	- Both the "Continuous" and "sounds" folders should be placed in the same directory as the .py files.


# How to run the code
1. Open a terminal and run, sequentially: 
	- py unsupervised.py
	- py supervised.py
	- py online.py

# Testing other datasets
1. Convert .dat file to .mat format, by running "file_open_vEEG_data" in the Matlab terminal. 
2. Save the .mat files in a folder named "Continuous".
3. Run previous code. 
