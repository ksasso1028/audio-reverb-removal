# Dereverb-audio
Code to train a custom time-domain autoencoder to dereverb audio. The SOX reverb algorithm was used to explore as a baseline before moving to impulse responses. 

Trained using pytorch 1.10.1 on a single RTX 3090 in a Ubuntu Workstation

# Install dependencies

Import the conda env file to a new environment 
```
conda env create -f deverb-env.yml -n envName
```

# Download dataset

Dataset used to train this model was the Divide and Remaster dataset introduced by Mistubishi.

Can download here https://zenodo.org/record/5574713

# Model Architecture

Model architecture can be found in dereverb/autoVerb.py. It is inpsired networks like Demucs and ConvTasNet

# Training

use trainReverb.py to train a model. You can configure hyperparemeters like epochs, sample rate, etc using parser arguments 
```
//example
python trainReverb.py modelName --epochs 1000000 -lr .0001 -b 16 -sec 2
```

A trainConfig will be generated in the configs folder saving various hyperparameters. This is to continue training in the event of a crash or to explore hyperparameters of trained models


Tensorboard is utilized to view model outputs during training and inspect train/test losses


# Evaluate

metrics.py is used to generate metrics relating to the L1 delta and SI-SNR of the output and ground truth audio file

# Test set METRICS
```
Average L1 Delta = .005
Average SISNR = 11.68
```

# Visualization
dereverb-webapp.py is a streamlit website to evaluate model outputs in real time with the ability to configure reverb parameters.


