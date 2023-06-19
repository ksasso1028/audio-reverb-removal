import torch
import streamlit as st
import glob
import torchaudio
from audio_dataset import loadSample
import os
import pandas
from dereverb.autoVerb import AutoVerb
import pandas as pd
import auraloss
@st.cache
def getDevice():
    device = "cuda"
    deviceCount = torch.cuda.device_count()
    # No GPUS available
    if deviceCount == 0:
        device = "cpu"
    trainDevice = torch.device(device + ":0")
    return trainDevice

@st.cache
def load(weights,trainDevice):
    net= AutoVerb(blocks=5, inChannels=48, channelFactor=48)
    net.load_state_dict(torch.load(weights))
    return net.to(trainDevice)

st.title("Reverb  testing")
# test set of cocktail fork, can change to your own dataset here (assuming it has a speech column).
audio = 'datasets/cocktail-fork-test.csv'

all = pd.read_csv(audio)
speech = all["speech"]

weights = "weights/"
trainDevice = getDevice()
models = []
for file in os.listdir(weights):
    if file.endswith('.pt'):
        w = weights+ file
        models.append(w)
go = None

l1 = torch.nn.L1Loss()
sisnr = auraloss.time.SISDRLoss()

with st.sidebar:
    mOption = st.selectbox("Choose a model to perform dereverberation  ", tuple(models), key="model-in")
    if mOption != None:
        net = load(mOption, trainDevice)
    option = st.selectbox("Choose an audio file to perform dereverberation! ",speech, key="audio-in")

    seconds = st.slider("Number of seconds", value=10, min_value=0, max_value=60,
                     key="db-slider-tanh")
    dry = loadSample(option, sampleRate=44100, length=seconds, segment=False, mono=False)
    #print(dry)

    verb = st.slider("Amount of verb", value=0, min_value=0, max_value=100, key="verb")
    stereo = st.slider("Amount of stereo", value=0, min_value=0, max_value=100,key="stereo")
    damp = st.slider("Amount HF dampning", value=0, min_value=0, max_value=100, key="HF-damp")
    roomScale = st.slider("Size of room", value=0, min_value=0, max_value=100,  key="room-scale")

    effects = [
        ["reverb", str(verb), str(damp), str(roomScale), str(stereo)],
    ]
    wet, sr = torchaudio.sox_effects.apply_effects_tensor(dry.clone(), 44100, effects)
    # fix clipping if present after verb alg
    if abs(wet).max() > 1:
        wet = (wet / (abs(wet).max())).clone()
    # add batch dimension for input
    wet = wet.to(trainDevice).unsqueeze(0)
    go = st.button("Process!", key="deverb")

if go:
    with torch.no_grad():
        out = net(wet)
        out = out.cpu()
        SISNR = sisnr(out, dry)
        l1Detla = l1(out, dry)

    st.write("L1 DELTA -> ",l1Detla.item())
    st.write("SISNR->", SISNR.item())


    og = "original.wav"
    w = "wet.wav"
    repair = "netProcessed.wav"
    # fix clipping in output if exists
    if abs(out).max() > 1:
        out = out / abs(out).max()

    torchaudio.save(og, dry.cpu().detach().squeeze(0), 44100)
    torchaudio.save(w, wet.cpu().detach().squeeze(0), 44100)
    clipped = wet[wet >= 1]
    torchaudio.save(repair, out.cpu().detach().squeeze(0), 44100)
    st.subheader("original")
    audio_file = open(og, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes)
    st.subheader("wet")
    audio_file = open(w, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes)
    st.subheader("repair")
    audio_file = open(repair, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes)



