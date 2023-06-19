import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-tr", "--train" ,help = "Specify name of train dataset csv (with .csv) ", type = str, default = "cocktail-fork-train.csv")
parser.add_argument("-tt", "--test" ,help = "Specify name of test dataset csv (with .csv)", type = str, default = "cocktail-fork-test.csv")
args = parser.parse_args()

train = ['/home/sasso/datasets-local/dnr/tr/']
test = ['/home/sasso/datasets-local/dnr/tt/']
header = ["music","sfx","annots","mix","speech "]

trainCsv = pd.DataFrame(columns=header)
testCsv = pd.DataFrame(columns=header)


for rootdir in train:
    for subdir, dirs, files in os.walk(rootdir):
        data = []
        for file in files:
            file = os.path.join(subdir, file)
            #print(file)
            data.append(file)
        print(data)
        if len(data) == 5:
            trainCsv  = trainCsv.append(pd.Series(data,index=trainCsv.columns), ignore_index=True)

for rootdir in test:
    for subdir, dirs, files in os.walk(rootdir):
        data = []
        for file in files:
            file = os.path.join(subdir, file)
            #print(file)
            data.append(file)
        print(data)
        if len(data) == 5:
            testCsv  = testCsv.append(pd.Series(data,index=testCsv.columns), ignore_index=True)

trainCsv.to_csv(args.train, index=False)
testCsv.to_csv(args.test, index=False)

