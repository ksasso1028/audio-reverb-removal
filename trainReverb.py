from audio_dataset import audioDatasetEffectsSpeech
from dereverb.autoVerb import AutoVerb
import os
import torch
import numpy as np
import time
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import argparse
import random
seed = 1028
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# When running on the CuDNN backend, two further options must be set
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Set a fixed value for the hash seed
os.environ["PYTHONHASHSEED"] = str(seed)

from stft.STFT_loss import MultiResolutionSTFTLoss
device = "cuda"
deviceCount = torch.cuda.device_count()
# No GPUS available
if  deviceCount == 0:
    device = "cpu"

parser = argparse.ArgumentParser()
# Required args
parser.add_argument("model", help = "Name of the model being trained")
# Optional args
parser.add_argument("-b", "--batchSize", help = "Specify size of batch for training set, default is 32", type = int, default = 32)
parser.add_argument("-cpu" ,"--cpu", help = "Flag to use cpu for training, default device is CUDA", action = "store_true")
parser.add_argument("-d", "--debug" ,help = "Flag to output loss at each training step for debugging worker count", action = "store_true")
parser.add_argument("-dt", "--dataType", help = "Data type for training, default is float32", default = "float")
parser.add_argument("-e", "--epochs", help = "Specify number of epochs, default is 50", type = int, default = 500000000)
parser.add_argument("-id", "--deviceID", help = "Specify which GPU to use, default is 0", type = int, default = 0)
parser.add_argument("-lr", "--learningRate" ,help = "Specify initial learning rate for training", type = float, default = .04)
parser.add_argument("-r", "--runDir" ,help = "Specify directory for tensorboard events", type = str, default = "runs/")
parser.add_argument("-rm", "--remove" ,help = "Flag to remove old Tensorboard folder for a model if it exists", action = "store_true")
parser.add_argument("-s", "--script", help = "Flag to save model as a script module", action = "store_true")
parser.add_argument("-w", "--workers" ,help = "Specify number of workers for training, default is 10", type = int, default = 1)
parser.add_argument("-sr", "--sampleRate" ,help = "Specify sample rate, default is 16k", type = int, default = 44100)
parser.add_argument("-sec", "--seconds", help = "Specify number of seconds to use for training/testing samples", type = float, default = 2)
args = parser.parse_args()

# Create model + config dirs for organization
if os.path.isdir("models"):
        print("models" + " already exists")
else:
    os.makedirs("models")

if os.path.isdir("configs"):
        print("configs" + " already exists")
else:
    os.makedirs("configs")

typeMap = {"float" : torch.float, "bfloat" : torch.bfloat16, "half" : torch.half, "double" :torch.double}

if args.cpu:
    device = "cpu"


if args.deviceID > deviceCount or args.deviceID < 0:
    print("INVALID DEVICE")
    # use last device
    args.deviceID = deviceCount - 1

# Set training device based on parser args
trainDevice = torch.device(device + ":" + str(args.deviceID))
if args.dataType in typeMap.keys():
    dataType = typeMap[args.dataType]
else:
    print(args.dataType + " IS NOT A VALID DATATYPE, using float32")
    dataType = typeMap["float"]
    args.dataType = "float"

dtTag = ""
# no tag is assumed to be of type float32
if args.dataType != "float":
    dtTag = "-" + args.dataType


print("TRAINING DEVICE: " + str(trainDevice))

print("DATA TYPE BEING USED: " +str(dataType))
modelName = args.model + dtTag
def main():
    best = 10000000000000
    if args.remove:
        if os.path.isdir(args.runDir + "/" + modelName):
            print("Removing old t-board events for " + modelName)
            os.system("rm -r " + args.runDir + "/" + modelName + "/")
            # delete may take some time
            time.sleep(7)

    writer = SummaryWriter(args.runDir + '/' + modelName)
    scheduler= None
    specLoss = MultiResolutionSTFTLoss().to(trainDevice, dtype=dataType)
    # configure CSV dataset
    train = audioDatasetEffectsSpeech(csv_file='datasets/cocktail-fork-train.csv', sample_rate =args.sampleRate, length=args.seconds, test=False, segment=True)
    test = audioDatasetEffectsSpeech(csv_file='datasets/cocktail-fork-test.csv', sample_rate = args.sampleRate, length=args.seconds,
                                     test=True, segment=False)
    trainLoader = DataLoader(train, batch_size =args.batchSize,
    num_workers= args.workers, shuffle = True)#, sampler = sampler )

    testLoader = DataLoader(test, batch_size =1,
            num_workers = args.workers ,shuffle=True)

    # Initialize custom network
    net = AutoVerb(blocks=5, inChannels=48, channelFactor=48)
    print(net)
    net = net.to(trainDevice, dtype=dataType)
    print("NUMBER OF PARAMETERS ,", sum(p.numel() for p in net.parameters()))
    net.load_state_dict(torch.load("weights/best_reverb-high-stft-2.pt"))
    l1 = nn.L1Loss()#asteroid.losses.sdr.SingleSrcNegSDR("sisdr")

    # Choose to use scheduler
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Optimizer for custom network
    optimizer = optim.Adam(net.parameters(), lr = args.learningRate)
    epochs = args.epochs
    print("Creating model training config...")

    # Save model training configuration in a text file incase we crash.
    modelCard = open("configs/trainConfig-" + modelName + ".txt", "w+")
    modelCard.write("MODEL CARD FOR : " + modelName + "\n")
    modelCard.write("DATA TYPE : " + args.dataType + "\n")
    modelCard.write("EPOCHS : " + str(args.epochs) + "\n")
    modelCard.write("BATCH SIZE USED : " + str(args.batchSize) + "\n")
    modelCard.write("OPTIMIZER :")
    modelCard.write(str(optimizer) + "\n")
    if scheduler != None:
        modelCard.write("SCHEDULER ")
        modelCard.write(str(scheduler) + "\n")
    modelCard.write("LAYERS :")
    modelCard.write(str(net) + "\n")
    modelCard.close()
    print(net)
    print("Beginning training cycle!")
    print("NUMBER OF PARAMETERS ,", sum(p.numel() for p in net.parameters()))
    for epoch in range(epochs):  # loop over the dataset multiple times
        net.train()
        runningLossSpeech = 0
        runningLossSpeechFFT = 0
        runningLossTotal = 0
        #if epoch == 5:
            #for g in optimizer.param_groups:
                #g['lr'] = 0.007
        for data in trainLoader:
            dry, wet = data['dry'], data['wet']

            optimizer.zero_grad()
            speech = net(wet.to(trainDevice, dtype=dataType))

            lossSpeech = l1(speech, dry.to(trainDevice, dtype=dataType))
            sc, lossSFFT = specLoss(speech.to(trainDevice, dtype=dataType),
                                   dry.to(trainDevice, dtype=dataType))

            loss =   lossSpeech + lossSFFT
            runningLossTotal += loss.item()
            loss.backward()
            runningLossSpeech += lossSpeech.item()
            runningLossSpeechFFT += lossSFFT.item()
            # print(runningLoss)
            optimizer.step()
            if args.debug:
                print("Train Loss", loss.item())
        averageLossSpeech= runningLossSpeech /len(trainLoader)
        averageLossSpeechFFT= runningLossSpeechFFT /len(trainLoader)
        averageLossTotal = runningLossTotal / len(trainLoader)
        #print("writing to tensorboard...")
        writer.add_scalar('training loss total', averageLossTotal, epoch)
        writer.add_scalar('training loss Speech (L1)', averageLossSpeech, epoch)
        writer.add_scalar('training loss Speech (MULTI-STFT)', averageLossSpeechFFT, epoch)
        writer.add_scalar('learning rate', float(optimizer.state_dict()['param_groups'][0]['lr']), epoch)
        net.eval()
        runningTestLossSpeech = 0
        runningTestLossSpeechFFT = 0
        runningTestLossTotal = 0
        one = 0
        two = 0
        three = 0
        print("running validation..")
        with torch.no_grad():
            for data in testLoader:
                dry, wet = data['dry'], data['wet']
                speech = net(wet.to(trainDevice, dtype=dataType))
                lossSpeech = l1(speech, dry.to(trainDevice, dtype=dataType))
                sc, lossSFFT = specLoss(speech.to(trainDevice, dtype=dataType),
                                       dry.to(trainDevice, dtype=dataType))
                runningTestLossSpeech +=  lossSpeech.item()
                runningTestLossSpeechFFT += lossSFFT.mean().item()
                runningTestLossTotal += lossSpeech.item() + lossSFFT.item()
                if args.debug:
                    print("Test Loss", lossSpeech.item())
                one = dry
                two = speech
                three = wet
        averageTestLossSpeech = runningTestLossSpeech / len(testLoader)
        averageTestLossSpeechFFT = runningTestLossSpeechFFT / len(testLoader)
        averageTestLossTotal = runningTestLossTotal / len(testLoader)
        # print("writing to tensorboard...")
        writer.add_scalar('test loss total', averageTestLossTotal, epoch)
        writer.add_scalar('test loss Speech (L1 )', averageTestLossSpeech, epoch)
        writer.add_scalar('test loss Speech (MULTI STFT)', averageTestLossSpeechFFT, epoch)
        # save best model during training
        if averageTestLossTotal < best:
            best = averageTestLossTotal
            torch.save(net.state_dict(), "models/best_" + modelName + ".pt")
            tracedModel = torch.jit.script(net)
            tracedModel.save("models/best_scripted_" + modelName + ".pt")
            print("BEST ACCURACY, SAVING MODEL")
        print("AVERAGE TEST LOSS (TOTAL) FOR EPOCH " + str(epoch) + ":  " + str(averageTestLossTotal))
        print("AVERAGE TEST LOSS SPEECH (L1) FOR EPOCH " + str(epoch) + ":  " + str(averageTestLossSpeech))
        print("AVERAGE TEST LOSS SPEECH (MULTI-STFT) FOR EPOCH " + str(epoch) + ":  " + str(averageTestLossSpeechFFT))
        # tensorboard only uploads mono audio
        writer.add_audio('sample audio (SPEECH) ', torch.mean(one, dim=1, keepdim=True), epoch,
                         sample_rate=args.sampleRate)
        writer.add_audio('sample audio (WET) ', torch.mean(three, dim=1, keepdim=True), epoch,
                         sample_rate=args.sampleRate)
        writer.add_audio('sample audio (REPAIR) ', torch.mean(two, dim=1, keepdim=True), epoch, sample_rate=args.sampleRate)
        #scheduler.step()
        print(epoch, "#####################################")
    print('Finished Training')
    torch.save(net.state_dict(), "models/" + modelName +  ".pt")
    if args.script:
        scriptedModel = torch.jit.script(net)
        scriptedModel.save("models/scripted_" + modelName + ".pt")
    writer.close()
if __name__=="__main__":
        #torch.set_num_threads(1)
        main()

