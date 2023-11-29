import torch
from dereverb.auto_verb import AutoVerb
from audio_dataset import AudioDatasetReverb
from torch.utils.data import  DataLoader
from stft.STFT_loss import MultiResolutionSTFTLoss
import auraloss

test = AudioDatasetReverb(csv_file='datasets/cocktail-fork-test.csv', sample_rate = 44100, length=20, test=True, segment=False)
test_loader = DataLoader(test, batch_size =1,num_workers = 5 ,shuffle=True)
train_device = torch.device("cuda:0")

spec_loss = MultiResolutionSTFTLoss().to(train_device)
net = AutoVerb(blocks=5, inChannels=48, channelFactor=48)
net.load_state_dict(torch.load("weights/best_reverb-high-stft-3.pt"))
net = net.to(train_device)
net = net.eval()
print("NUMBER OF PARAMETERS ,", sum(p.numel() for p in net.parameters()))
l1 = torch.nn.L1Loss().to(train_device)
sisnr = auraloss.time.SISDRLoss().to(train_device)

running_test_l1 = 0
running_test_stft = 0
running_test_sisnr = 0

# keep track of SISNR nans
nans = 0
with torch.no_grad():
    for data in test_loader:
        dry, wet = data['dry'], data['wet']
        speech = net(wet.to(train_device))
        loss_l1 = l1(speech, dry.to(train_device))
        sc, loss_stft = spec_loss(speech.to(train_device),dry.to(train_device))
        loss_sisnr = sisnr(speech.to(train_device), dry.to(train_device))
        print(loss_sisnr)
        print(loss_stft)
        running_test_l1 += loss_l1.cpu().item()
        running_test_stft += loss_stft.cpu().item()
        if torch.isnan(loss_sisnr):
            nans += 1
        else:
            running_test_sisnr +=  loss_sisnr.cpu().item()
        print("#############################################")

average_test_l1 = running_test_l1 / len(test_loader)
average_test_stft = running_test_stft / len(test_loader)
# remove nans from average calculation
average_test_sisnr = running_test_sisnr / (len(test_loader) - nans)

print("AVERAGE l1 LOSS ->  ", average_test_l1)
print("AVERAGE STFT LOSS-> ", average_test_stft)
print("AVERAgE SISNR -> ", average_test_sisnr)


