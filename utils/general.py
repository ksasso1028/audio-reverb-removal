import os
import torch
import random
import numpy as np
import torchaudio


def create_model_card(net, name, data_type, batch_size, epochs, opt):
    print("Creating model training config...")
    # Save model training configuration in a text file incase we crash.
    model_card = open("configs/trainConfig-" + name + ".txt", "w+")
    model_card.write("MODEL CARD FOR : " + name + "\n")
    model_card.write("DATA TYPE : " + str(data_type) + "\n")
    model_card.write("EPOCHS : " + str(epochs) + "\n")
    model_card.write("BATCH SIZE USED : " + str(batch_size) + "\n")
    model_card.write("OPTIMIZER :")
    model_card.write(str(opt) + "\n")
    model_card.write("LAYERS :")
    model_card.write(str(net) + "\n")
    model_card.close()
    print(net)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

# loads the raw samples of an audio file either in stereo or Mono, optional random segmenting for training
def load_sample(mix, sample_rate,load=True ,mono=True, length = 120, resample = True, segment=True):
    # loading mix from disk
    if load:
        metadata = torchaudio.info(mix)
        sr = metadata.sample_rate
        # will load entire audio file we do not segment
        if segment:
            start = random.randint(0, ((metadata.num_frames) - 1))
            # use original sr
            num_frames = int(length * sr)
            wav, ss = torchaudio.load(mix, frame_offset=start,num_frames=num_frames)
        else:
            # load all
            wav, ss = torchaudio.load(mix)
        #print(wav.shape)
        if resample:
            resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
            wav_file = resample(wav).clone()
        else:
            wav_file = wav

        if mono:
            wav_file = torch.mean(wav_file, dim=0, keepdim=True)

        # stack wav on both channels if mono
        elif wav_file.size(0) < 2:
            wav_file = torch.cat([wav_file,wav_file.clone()], dim=0)
        # only retrieve the first 2 channels if more exist
        else:
            wav_file = wav_file[:2, :]

    # mix is a tensor, already loaded
    else:
        wav_file = mix
    #print(wav_file.shape)
    #wav_file = mono.clone()

    if length > 0:
    # print(four)
        len = int(sample_rate * length)
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

