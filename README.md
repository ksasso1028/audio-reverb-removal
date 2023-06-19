# Dereverb-audio
Code to train a custom time-domain autoencoder to dereverb audio. The SOX reverb algorithm was used to explore as a baseline before moving to impulse responses. 

Trained using pytorch 1.10.1 on a single RTX 3090 in a Ubuntu Workstation

# Severe Examples

```
Dry 
```
https://github.com/ksasso1028/Dereverb-audio/assets/11267645/49667381-8477-4d0a-ac8d-91c4df77fe75


```
Wet
```

https://github.com/ksasso1028/Dereverb-audio/assets/11267645/ec9ad908-677a-4de7-b315-5a0c298e78b2


```
autoencoder repair (dereverb)
```

https://github.com/ksasso1028/Dereverb-audio/assets/11267645/66bb6960-a3ae-4893-aa2b-e6861b78ce1e



# Medium Examples

```
Dry 
```


https://github.com/ksasso1028/Dereverb-audio/assets/11267645/da7ca6c0-d549-41ab-a576-a3bc2b782b1b



```
Wet
```


https://github.com/ksasso1028/Dereverb-audio/assets/11267645/50f01d0f-62dd-4b07-8541-10fc0ed9df8c



```
autoencoder repair (dereverb)
```



https://github.com/ksasso1028/Dereverb-audio/assets/11267645/484c7782-af1d-4de2-a1f6-9a4080b5341a



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
Average SISNR = 11.68db
```

# Visualization
dereverb-webapp.py is a streamlit website to evaluate model outputs in real time with the ability to configure reverb parameters.


