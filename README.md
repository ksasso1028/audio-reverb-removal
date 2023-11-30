# Audio Reverb Removal with Pytorch
Code to train a custom time-domain autoencoder to dereverb audio. The SOX reverb algorithm was used to explore as a baseline before moving to impulse responses.

### Will not run on windows, as its not SOX compatible. Linux only

Trained using pytorch 1.10.1 on a single RTX 3090 in a Ubuntu Workstation.


**Examples can be heard at the bottom of this README**


**Designed for Speech, CD-Quality (44.1khz)**



# Install dependencies

Import the conda env file to a new environment 
```
conda env create -f deverb-env.yml -n envName
```

# Download dataset

Dataset used to train this model was the Divide and Remaster dataset introduced by Mistubishi.

Can download here https://zenodo.org/record/5574713

# Model Architecture

Model architecture can be found in dereverb/auto_verb.py. It is a custom time domain denoising autoencoder inspired by Demucs and ConvTasNet

# Training

use train_reverb.py to train a model. You can configure hyperparemeters like epochs, sample rate, etc using parser arguments 
```
//example
python trainReverb.py modelName --epochs 1000000 -lr .0001 -b 16 -sec 2
```

***Model was optimized using a L1 loss, along with a Multi-res STFT adapted from the CleanUNET paper***


[CleanUnet](https://github.com/NVIDIA/CleanUNet)

A trainConfig will be generated in the configs folder saving various hyperparameters. This is to continue training in the event of a crash or to explore hyperparameters of trained models


Tensorboard is utilized to view model outputs during training and inspect train/test losses


# Evaluate

metrics.py is used to generate metrics relating to the L1 delta and SI-SNR of the output and ground truth audio file

# Test set METRICS
```
Average L1 Delta = .005
Average SISNR = 11.68db
```

# Visualization
dereverb_webapp.py is a streamlit website to evaluate model outputs in real time with the ability to configure reverb parameters.

