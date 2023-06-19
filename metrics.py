import torch
from dereverb.autoVerb import AutoVerb
from audio_dataset import audioDatasetEffectsSpeech
from torch.utils.data import  DataLoader
from stft.STFT_loss import MultiResolutionSTFTLoss
import auraloss

test = audioDatasetEffectsSpeech(csv_file='datasets/cocktail-fork-test.csv', sample_rate = 44100, length=20, test=True, segment=False)
testLoader = DataLoader(test, batch_size =1,num_workers = 5 ,shuffle=True)
trainDevice = torch.device("cuda:0")

specLoss = MultiResolutionSTFTLoss().to(trainDevice)
net= AutoVerb(blocks=5, inChannels=48, channelFactor=48)
net.load_state_dict(torch.load("weights/best_reverb-high-stft-3.pt"))
net = net.to(trainDevice)
net = net.eval()
print("NUMBER OF PARAMETERS ,", sum(p.numel() for p in net.parameters()))
l1 = torch.nn.L1Loss().to(trainDevice)
sisnr = auraloss.time.SISDRLoss().to(trainDevice)

runningTestLossL1 = 0
runningTestLossSTFT = 0
runningTestLossSISNR = 0

# keep track of SISNR nans
nans = 0
with torch.no_grad():
    for data in testLoader:
        dry, wet = data['dry'], data['wet']
        speech = net(wet.to(trainDevice))
        lossSpeech = l1(speech, dry.to(trainDevice))
        sc, lossSFFT = specLoss(speech.to(trainDevice),dry.to(trainDevice))
        lossSISNR = sisnr(speech.to(trainDevice), dry.to(trainDevice))
        print(lossSISNR)
        print(lossSFFT)
        runningTestLossL1 += lossSpeech.cpu().item()
        runningTestLossSTFT += lossSFFT.cpu().item()
        if torch.isnan(lossSISNR):
            nans += 1
        else:
            runningTestLossSISNR +=  lossSISNR.cpu().item()
        print("#############################################")

averageTestLossL1 = runningTestLossL1/ len(testLoader)
averageTestLossSpeechSTFT = runningTestLossSTFT/ len(testLoader)
# remove nans from average calculation
averageTestLossSISNR = runningTestLossSISNR / (len(testLoader) - nans)

print("AVERAGE l1 LOSS ->  ", averageTestLossL1)
print("AVERAGE STFT LOSS-> ", averageTestLossSpeechSTFT)
print("AVERAgE SISNR -> ", averageTestLossSISNR)


