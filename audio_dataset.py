from __future__ import print_function, division
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import random
from torch_audiomentations import Compose, LowPassFilter, HighPassFilter, AddColoredNoise, PitchShift
import torch
class audioDatasetEffectsSpeech(Dataset):

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
        dry = loadSample(audio,sampleRate=self.sample_rate, length = self.len, segment=self.segment, mono=False)
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

# loads the raw samples of an audio file either in stereo or Mono, optional random segmenting for training
def loadSample(mix, sampleRate,load=True ,mono=True, length = 120, resample = True, segment=True):
    # image name contains full relative path so no need to use rootdir.
    if load:
        metadata = torchaudio.info(mix)
        sr = metadata.sample_rate
        if segment:
            start = random.randint(0, ((metadata.num_frames) - 1))
            num_frames = int(length * sr)
            wav, ss = torchaudio.load(mix, frame_offset=start,num_frames=num_frames)
        else:
            # load all
            wav, ss = torchaudio.load(mix)
        #print(wav.shape)
        if resample:
            resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampleRate)
            wav_file = resample(wav).clone()
        else:
            wav_file = wav

        if mono:
            wav_file = torch.mean(wav_file, dim=0, keepdim=True)

        elif wav_file.size(0) < 2:
            wav_file = torch.cat([wav_file,wav_file.clone()], dim=0)
        else:
            wav_file = wav_file[:2, :]
    else:
        wav_file = mix
    #print(wav_file.shape)
    #wav_file = mono.clone()

    if length > 0:
    # print(four)
        len = int(sampleRate * length)
    #print(len)
        # crop and pad
        if wav_file.size(1) < len:
        # print("LESS")
            pad = (0, len - wav_file.size(1))
            # print(pad)
            wav_file = torch.nn.functional.pad(wav_file.clone(), pad)
        else:
            wav_file = wav_file[:,:len]
    return wav_file


