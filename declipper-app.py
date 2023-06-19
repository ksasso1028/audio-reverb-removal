import torch
import streamlit as st
import glob
import torchaudio
from utils import load, loadSample,hardClip, harmonicOne, harmonicTwo, overdriveHarmonic, overdriveHard, overdriveHarmonicSoft, tanOverdrive,softTanh, arcSoftClip
import os



st.title("Declipper testing")

print(glob.glob("src-audio/*.wav"))
audio = "src-audio/"
allModels = "v2/"

files = []

for file in os.listdir(audio):
    if file.endswith('.wav') or file.endswith('.mp3'):
        w = audio + file
        files.append(w)
models = []
for file in os.listdir(allModels):
    if file.endswith('.pt'):
        w = allModels + file
        models.append(w)
go = None
with st.sidebar:
    mOption = st.selectbox("Choose a model to perform declipping ", tuple(models), key="model-in")
    if mOption != None:
        declipper = load(mOption)
        #declipper.moveDevice("cpu",1)
    option = st.selectbox("Choose an audio file to perform declipping! ", tuple(files), key="audio-in")

    # distortion params to change type of distortion, etc.
    # if declip button pressed!
    seconds = st.slider("Number of seconds", value=10, min_value=0, max_value=60,
                     key="db-slider-tanh")
    dry = loadSample(option, True, 44100, mono=True, length=seconds).squeeze(1)
    print(dry)
    clipTypes = ["Hard clip", "Harmonic Dist-1", "Harmonic Dist-2", "Overdrive Harmonics", "Hard clip, overdrive",
                 "Tanh Overdrive", "Pure Soft clip (tanh)", "Pure Soft clip (arctan)"]
    clipperOption = st.selectbox("Select a type of clipping to perform on audio", tuple(clipTypes), key="clipper")
    wet = None

    if clipperOption == "Hard clip":
        gain = st.slider("Amount of gain to add to audio (db)", value=20, min_value=0, max_value=30, key="db-slider")
        wet= hardClip(dry, gain)
        print(wet.shape)

    if clipperOption == "Pure Soft clip (tanh)":
        gain = st.slider("Amount of gain to add to audio (db)", value=20, min_value=0, max_value=60,
                         key="db-slider-tanh")
        wet = softTanh(dry, gain)
        print(wet.shape)
    if clipperOption == "Tanh Overdrive":
        gain = st.slider("Amount of gain to add to audio (db)", value=0, min_value=0, max_value=60,
                         key="db-slider-tanh-o")
        h1 = st.slider("Amount of h1 harmonics", value=0.0, min_value=0.0, max_value=7.0, step=.01,
                       key="db-slider-tanh-1")
        h2 = st.slider("Amount h2 harmonics", value=0.0, min_value=0.0, max_value=7.0, step=.01, key="db-slider-tanh-2")
        color = st.slider("Amount of color for overdrive", value=0, min_value=0, max_value=100, key="db-slider-tanh-3")
        wet = overdriveHarmonicSoft(dry, h1, h2, gain, color)
    go = st.button("Declip!", key="declip")
if go:
    #wet = wet / abs(wet).max()
    out = declipper.predict(wet)
    #declipper.moveDevice("cpu",1)
    declipper.model.save(mOption)
    print(dry.max())
    out = out.cpu()

    if abs(dry).max() > 1:
        dry = dry / abs(dry).max()
    if abs(wet).max() > 1:
        wet = wet / abs(wet).max()

    og = "original.wav"
    clip = "clipped.wav"
    repair = "repaired.wav"
    if abs(out).max() > 1:
        out = out / abs(out).max()
    torchaudio.save(og, dry.cpu().detach(), 44100)
    torchaudio.save(clip, wet.cpu().detach().squeeze(0), 44100)
    clipped = wet[wet >= 1]
    print("% of clipped samples! ", len(clipped.flatten())/ len(wet.flatten()))
    torchaudio.save(repair, out.cpu().detach().squeeze(0), 44100)
    st.subheader("original")
    audio_file = open(og, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes)
    st.subheader("clipped")
    audio_file = open(clip, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes)
    st.subheader("repair")
    audio_file = open(repair, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes)
    print(out)



