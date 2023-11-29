import argparse
import os
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("-tr", "--train" ,help = "Specify name of train dataset csv (with .csv) ", type = str, default = "cocktail-fork-train.csv")
parser.add_argument("-tt", "--test" ,help = "Specify name of test dataset csv (with .csv)", type = str, default = "cocktail-fork-test.csv")
args = parser.parse_args()

train = ["D:\\dnr\\tr\\"]
test = ["D:\\dnr\\cv\\"]
header = ["annots","mix","music","sfx","speech"]

train_csv = pd.DataFrame(columns=header)
test_csv = pd.DataFrame(columns=header)


for rootdir in train:
    for subdir, dirs, files in os.walk(rootdir):
        data = []
        for file in files:
            file = os.path.join(subdir, file)
            #print(file)
            data.append(file)
        print(data)
        if len(data) == 5:
            train_csv  = train_csv._append(pd.Series(data,index=train_csv.columns), ignore_index=True)

for rootdir in test:
    for subdir, dirs, files in os.walk(rootdir):
        data = []
        for file in files:
            file = os.path.join(subdir, file)
            #print(file)
            data.append(file)
        print(data)
        if len(data) == 5:
            test_csv  = test_csv._append(pd.Series(data,index=test_csv.columns), ignore_index=True)

train_csv.to_csv(args.train, index=False)
test_csv.to_csv(args.test, index=False)

