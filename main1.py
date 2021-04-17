import os,sys
import pandas as pd
import numpy as np
import argparse
from torch import nn
import torch
from torch import tensor
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import numpy as np
from keras.utils.np_utils import to_categorical   
from sklearn.model_selection import train_test_split

from model import *
from accuracy import get_accuracy

# Reproducibility
torch.manual_seed(42)

models = {
    'classifier1': Classifier1,
    'classifier2': Classifier2,
    'classifier3': Classifier3,
    'classifier4': Classifier4
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='Water Quality Analysis')

    # Add more arguments based on requirements later
    parser.add_argument('-e', '--epochs', help='Set number of train epochs', default=500, type=int)
    parser.add_argument('-model', '--model', help='Set Feature extractor', default='classifier1', type=str)
    parser.add_argument('-lr', '--learning_rate', help='Set starting learning rate', default=0.1, type=float)
    parser.add_argument('-tts', '--train_test_split', help='Set fraction of dataset to be used for testing', default=0.45, type=float)

    parser.set_defaults(train=True)

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_arguments()
    print("Arguments: ", args)
    #########################################################
    ## DATA PRE PROCESSING
    #########################################################
    df = pd.read_excel (r'water-quality.xlsx')
    df = df[['ID', 'SITENAME', 'Mg', 'PH', 'K(Potassium)', 'NITRATE', 'SULPHATE',
        'EC(Electrical Conductivity)', 'Ca(Calcium)', 'Na(Sodium)', 'CARBONATE',
        'BICARBONATE', 'CHLORIDE', 'FLUORIDE', 'SAR(Sodium Absorption Ratio)',
        'RSC(Residual Sodium Carbonate', 'water Quality(A/B/C/D/E)']]

    X = df.drop(['ID', 'SITENAME', 'water Quality(A/B/C/D/E)'], axis = 1).values
    y = df['water Quality(A/B/C/D/E)'].values
    # Normalization of data
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std

    # Converting class labels into numerical form 0-4
    y = np.unique(y, return_inverse=True)[1].tolist()
    y = np.asarray(y)

    #########################################################
    ## SPLITTING DATA INTO TRAINING SET AND TESTING SET
    #########################################################

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.train_test_split, random_state=42, stratify = y)
    print(X_train.shape, X_test.shape)

    #########################################################
    ## MODEL SETTINGS
    #########################################################

    # Initializing the neural network classifier model
    classifier = models[args.model]()
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    # Converting the class labels into one-hot encoding
    y_train = to_categorical(y_train, num_classes=5)

    # Converting the data and targets into torch variables
    x_data = Variable(torch.from_numpy(X_train))
    y_data = Variable(torch.from_numpy(y_train))

    #########################################################
    ## TRAINING
    #########################################################
    for epoch in range(args.epochs):
        y_pred = classifier(x_data.float())
        l = criterion(y_pred, y_data)
        optimizer.zero_grad()
        if epoch % (args.epochs/10) == 0: # Print 10 updates
            print("Epoch: ", epoch, "Loss = ", l.item())
        l.backward()
        optimizer.step()

    pred_1 = classifier(x_data[0].float())
    print("Prediction for 1st data sample:",  pred_1)

    #########################################################
    ## TESTING
    #########################################################
    # Calculating accuracy by testing on the entire dataset

    x_test = Variable(torch.from_numpy(X_test))

    y_real = y_test
    output = classifier(x_test.float())
    y_pred = []
    for y in output:
        index_max = np.argmax(y.detach().numpy())
        y_pred.append(index_max)

    accuracy = get_accuracy(y_pred, y_real)
    print("Accuracy = {}%".format(accuracy))
