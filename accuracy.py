import os, sys
import numpy as np

def get_accuracy(y_pred, y_real):
    count = 0
    for i in range(len(y_real)):
        if y_pred[i] == y_real[i]:
            count = count + 1

    accuracy = (count / len(y_real)) * 100
    return accuracy