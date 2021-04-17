import pandas as pd
import numpy as np
from torch import nn
import torch
from torch import tensor
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import numpy as np
from keras.utils.np_utils import to_categorical   

from model import QualityClassifier
from accuracy import get_accuracy


if __name__ == "__main__":

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
    ## MODEL SETTINGS
    #########################################################

    # Initializing the neural network classifier model
    classifier = QualityClassifier()
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9)

    # Converting the class labels into one-hot encoding
    y = to_categorical(y, num_classes=5)

    # Converting the data and targets into torch variables
    x_data = Variable(torch.from_numpy(X))
    y_data = Variable(torch.from_numpy(y))

    #########################################################
    ## TRAINING
    #########################################################
    for epoch in range(500):
        y_pred = classifier(x_data.float())
        l = criterion(y_pred, y_data)
        optimizer.zero_grad()
        if epoch % 100 == 0:
            print("Epoch: ", epoch, "Loss = ", l.item())
        l.backward()
        optimizer.step()

    pred_1 = classifier(x_data[0].float())
    print("Prediction for 1st data sample:",  pred_1)

    #########################################################
    ## TESTING
    #########################################################
    # Calculating accuracy by testing on the entire dataset

    y_real = df['water Quality(A/B/C/D/E)'].values
    y_real = np.unique(y_real, return_inverse=True)[1].tolist()
    output = classifier(x_data.float())
    y_pred = []
    for y in output:
        index_max = np.argmax(y.detach().numpy())
        y_pred.append(index_max)

    accuracy = get_accuracy(y_pred, y_real)
    print("Accuracy = {}%".format(accuracy))
