from audio_dataset import AudioDatasetReverb
from dereverb.auto_verb import AutoVerb
import os
import torch
import time
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from stft.STFT_loss import MultiResolutionSTFTLoss
from utils.general import create_model_card, set_seed
import argparse


set_seed(1028)
device = "cuda"
device_count = torch.cuda.device_count()
# No GPUS available
if  device_count == 0:
    device = "cpu"

parser = argparse.ArgumentParser()
# Required args
parser.add_argument("model", help = "Name of the model being trained")
# Optional args
parser.add_argument("-b", "--batch_size", help = "Specify size of batch for training set, default is 32", type = int, default = 32)
parser.add_argument("-cpu" ,"--cpu", help = "Flag to use cpu for training, default device is CUDA", action = "store_true")
parser.add_argument("-d", "--debug" ,help = "Flag to output loss at each training step for debugging worker count", action = "store_true")
parser.add_argument("-dt", "--data_type", help = "Data type for training, default is float32", default = "float")
parser.add_argument("-e", "--epochs", help = "Specify number of epochs, default is 50", type = int, default = 500000000)
parser.add_argument("-id", "--device_id", help = "Specify which GPU to use, default is 0", type = int, default = 0)
parser.add_argument("-lr", "--learning_rate" ,help = "Specify initial learning rate for training", type = float, default = .04)
parser.add_argument("-r", "--run_dir" ,help = "Specify directory for tensorboard events", type = str, default = "runs/")
parser.add_argument("-rm", "--remove" ,help = "Flag to remove old Tensorboard folder for a model if it exists", action = "store_true")
parser.add_argument("-s", "--script", help = "Flag to save model as a script module", action = "store_true")
parser.add_argument("-w", "--workers" ,help = "Specify number of workers for training, default is 10", type = int, default = 1)
parser.add_argument("-sr", "--sample_rate" ,help = "Specify sample rate, default is 16k", type = int, default = 44100)
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

type_map = {"float" : torch.float, "bfloat" : torch.bfloat16, "half" : torch.half, "double" :torch.double}

if args.cpu:
    device = "cpu"


if args.device_id > device_count or args.device_id < 0:
    print("INVALID DEVICE")
    # use last device
    args.device_id = device_count - 1

# Set training device based on parser args
train_device = torch.device(device + ":" + str(args.device_id))
if args.data_type in type_map.keys():
    data_type = type_map[args.data_type]
else:
    print(args.data_type + " IS NOT A VALID DATATYPE, using float32")
    data_type = type_map["float"]
    args.data_type = "float"

dt_tag = ""
# no tag is assumed to be of type float32
if args.data_type != "float":
    dt_tag = "-" + args.data_type


print("TRAINING DEVICE: " + str(train_device))

print("DATA TYPE BEING USED: " +str(data_type))
model_name = args.model + dt_tag
def main():
    best = 10000000000000
    if args.remove:
        if os.path.isdir(args.runDir + "/" + model_name):
            print("Removing old t-board events for " + model_name)
            os.system("rm -r " + args.runDir + "/" + model_name + "/")
            # delete may take some time
            time.sleep(7)

    writer = SummaryWriter(args.run_dir + '/' + model_name)
    spec_loss = MultiResolutionSTFTLoss().to(train_device, dtype=data_type)
    # configure CSV dataset
    train = AudioDatasetReverb(csv_file='datasets/cocktail-fork-train.csv', sample_rate =args.sample_rate, length=args.seconds, test=False, segment=True)
    test = AudioDatasetReverb(csv_file='datasets/cocktail-fork-test.csv', sample_rate = args.sample_rate, length=args.seconds,
                                     test=True, segment=False)
    train_loader = DataLoader(train, batch_size =args.batch_size,
    num_workers= args.workers, shuffle = True)#, sampler = sampler )

    test_loader = DataLoader(test, batch_size =1,
            num_workers = args.workers ,shuffle=True)

    # Initialize custom network
    net = AutoVerb(blocks=5, in_channels=48, channel_factor=48)
    print(net)
    net = net.to(train_device, dtype=data_type)
    print("NUMBER OF PARAMETERS ,", sum(p.numel() for p in net.parameters()))
    #net.load_state_dict(torch.load("weights/best_reverb-high-stft-2.pt"))
    l1 = nn.L1Loss()

    # Optimizer for custom network
    optimizer = optim.Adam(net.parameters(), lr = args.learning_rate)
    epochs = args.epochs
    # create model card incase we crash.
    create_model_card(net=net,name=model_name,data_type=data_type,batch_size=args.batch_size,
                      epochs=args.epochs,opt=optimizer)
    print("Beginning training cycle!")
    print("NUMBER OF PARAMETERS ,", sum(p.numel() for p in net.parameters()))
    for epoch in range(epochs):  # loop over the dataset multiple times
        net.train()
        running_loss_l1 = 0
        running_loss_stft = 0
        running_loss_total = 0
        #if epoch == 5:
            #for g in optimizer.param_groups:
                #g['lr'] = 0.007
        for data in train_loader:
            dry, wet = data['dry'], data['wet']

            optimizer.zero_grad()
            speech = net(wet.to(train_device, dtype=data_type))

            loss_l1 = l1(speech, dry.to(train_device, dtype=data_type))
            sc, loss_stft = spec_loss(speech.to(train_device, dtype=data_type),
                                   dry.to(train_device, dtype=data_type))
            # multi task learning
            loss = loss_l1 + loss_stft
            running_loss_total += loss.item()
            loss.backward()

            running_loss_l1 += loss_l1.item()
            running_loss_stft += loss_stft.item()
            optimizer.step()
            if args.debug:
                print("Train Loss", loss.item())
        average_loss_l1 =  running_loss_l1 /len(train_loader)
        average_loss_stft = running_loss_stft /len(train_loader)
        average_loss_total = running_loss_total / len(train_loader)
        #print("writing to tensorboard...")
        writer.add_scalar('training loss total', average_loss_total, epoch)
        writer.add_scalar('training loss Speech (L1)', average_loss_l1, epoch)
        writer.add_scalar('training loss Speech (MULTI-STFT)', average_loss_stft, epoch)
        writer.add_scalar('learning rate', float(optimizer.state_dict()['param_groups'][0]['lr']), epoch)
        net.eval()
        running_test_l1 = 0
        running_test_stft = 0
        running_test_total = 0
        ground_truth = 0
        net_repair = 0
        reverbed = 0
        print("running validation..")
        with torch.no_grad():
            for data in test_loader:
                dry, wet = data['dry'], data['wet']
                speech = net(wet.to(train_device, dtype=data_type))
                loss_l1 = l1(speech, dry.to(train_device, dtype=data_type))
                sc, loss_stft = spec_loss(speech.to(train_device, dtype=data_type),
                                       dry.to(train_device, dtype=data_type))
                running_test_l1 +=  loss_l1.item()
                running_test_stft  += loss_stft.item()
                running_test_total += loss_l1.item() + loss_stft.item()
                if args.debug:
                    print("Test Loss", loss_l1.item())
                ground_truth = dry
                net_repair = speech
                reverbed = wet
        average_test_l1 = running_test_l1 / len(test_loader)
        average_test_stft = running_test_stft  / len(test_loader)
        average_test_total = running_test_total / len(test_loader)
        # print("writing to tensorboard...")
        writer.add_scalar('test loss total', average_test_total, epoch)
        writer.add_scalar('test loss Speech (L1 )', average_test_l1, epoch)
        writer.add_scalar('test loss Speech (MULTI STFT)', average_test_stft, epoch)
        # save best model during training
        if average_test_total < best:
            best = average_test_total
            torch.save(net.state_dict(), "models/best_" + model_name + ".pt")
            traced_model = torch.jit.script(net)
            traced_model.save("models/best_scripted_" + model_name + ".pt")
            print("BEST ACCURACY, SAVING MODEL")
        print("AVERAGE TEST LOSS (TOTAL) FOR EPOCH " + str(epoch) + ":  " + str(average_test_total))
        print("AVERAGE TEST LOSS SPEECH (L1) FOR EPOCH " + str(epoch) + ":  " + str(average_test_l1))
        print("AVERAGE TEST LOSS SPEECH (MULTI-STFT) FOR EPOCH " + str(epoch) + ":  " + str(average_test_stft))
        # tensorboard only uploads mono audio
        writer.add_audio('sample audio (SPEECH) ', torch.mean(ground_truth, dim=1, keepdim=True), epoch,
                         sample_rate=args.sample_rate)
        writer.add_audio('sample audio (WET) ', torch.mean(reverbed, dim=1, keepdim=True), epoch,
                         sample_rate=args.sample_rate)
        writer.add_audio('sample audio (REPAIR) ', torch.mean(net_repair, dim=1, keepdim=True), epoch, sample_rate=args.sample_rate)
        print(epoch, "#####################################")
    print('Finished Training')
    torch.save(net.state_dict(), "models/" + model_name +  ".pt")
    if args.script:
        scripted_model = torch.jit.script(net)
        scripted_model.save("models/scripted_" + model_name + ".pt")
    writer.close()
if __name__=="__main__":
        #torch.set_num_threads(1)
        main()

