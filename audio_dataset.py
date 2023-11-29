from __future__ import print_function, division
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import random
from utils.general import load_sample
from torch_audiomentations import Compose, PitchShift
import torch
class AudioDatasetReverb(Dataset):
    def __init__(self, csv_file, sample_rate,length,test,segment):
        self.frame = pd.read_csv((csv_file),engine='python')
        self.sample_rate = sample_rate
        torchaudio.set_audio_backend("sox_io")
        self.len = length
        self.test = test
        self.segment = segment


    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        audio =self.frame.loc[idx, 'speech']

        # if segment is true, it will randomly slice the audio somewhere in time and zero pad the rest if needed
        dry = load_sample(audio,sample_rate=self.sample_rate, length = self.len, segment=self.segment, mono=False)
        #start = random.randint(0, int(d.size(0) - (self.len * self.sample_rate)))
        # experimental modeling with sox reverb, before moving to impulse responses.
        verb = random.randint(10,95)
        stereo = random.randint(0,100)
        hf = random.randint(0,100)
        roomScale = random.randint(0,100)
        effects = [
            ["reverb", str(verb), str(hf), str(roomScale), str(stereo)],
        ]
        augment = Compose(
            transforms=[
                PitchShift(sample_rate=self.sample_rate, min_transpose_semitones=-6, max_transpose_semitones=6, p=.15)
            ]
        )
        if not self.test:
            # pitch shift for data augmentation only  on training set
            dry = augment(samples=dry.unsqueeze(0), sample_rate=self.sample_rate).squeeze(0)

        # create wet signal
        wet, sr = torchaudio.sox_effects.apply_effects_tensor(
            dry.clone(), self.sample_rate, effects)
        # fix clipping if present
        if abs(wet).max() > 1:
            wet = (wet / (abs(wet).max())).clone()
        sample = {"dry":dry, "wet":wet}
        return sample


