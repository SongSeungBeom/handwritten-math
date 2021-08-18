from keras.utils import np_utils
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.layers import *
from sklearn.metrics import accuracy_score
from tqdm import tqdm

path = r'C:\Users\Song\Desktop\myproject\dataset\\'
trainFileList = os.listdir(path+'train')
testFileList = os.listdir(path+'test')

trainFileList = [x for x in trainFileList if 'jpg' in x]
testFileList = [x for x in testFileList if 'jpg' in x]

word = {"!" : 0, "(" : 1, ")" : 2, "+" : 3, "," : 4, "-" : 5, "0" : 6, "1" : 7, "2" : 8, "3" : 9, "4" : 10, "5" : 11, "6" : 12, "7" : 13, "8" : 14, "9" : 15, "=" : 16,
        "[" : 17, "]" : 18, "A" : 19, "alpha" : 20, "b" : 22, "beta" : 23, "C" : 24, "cos" : 25, "d" : 26, "Delta" : 27, "div" : 28, "e" : 29, "f" : 30,
        "forward_slash" : 31, "G" : 32, "gamma" : 33, "geq" : 34, "gt" : 35, "H" : 36, "i" : 37, "in" : 38, "infty" : 39, "int" : 40, "j" : 41, "k" : 42, "l" : 43, "lambda" : 44,
        "ldots" : 45, "leq" : 46, "lim" : 47, "log" : 48, "lt" : 49, "M" : 50, "mu" : 51, "N" : 52, "neq" : 53, "o" : 54, "p" : 55, "phi" : 56, "pi" : 57, "pm" : 58, "q" : 59, "R" : 60,
        "rightarrow" : 61, "S" : 62, "sigma" : 63, "sin" : 64, "sqrt" : 65, "sum" : 66, "T" : 67, "tan" : 68, "theta" : 69, "times" : 70, "u" : 71, "v" : 72, "w" : 73, "X" : 74, "y" : 75,
        "z" : 76, "{" : 77, "}" : 78}

for i, file in enumerate(tqdm(trainFileList)):
    label, _ = file.split(' ')
    code, ext = _.split('.')
    code = code.replace("(", "").replace(")", "")

    new_img = load_img(path+'train\\'+file)
    arr_img = img_to_array(new_img)
    img = arr_img.reshape((1,)+arr_img.shape)

    if i == 0 :
        container = img
        labels = word[label]
    else:
        container = np.vstack([container, img])
        labels = np.vstack([labels, word[label]])

xTrain = container
yTrain = np_utils.to_categorical(labels,79)

for i, file in enumerate(tqdm(testFileList)):
    label, _ = file.split(' ')
    code, ext = _.split('.')
    code = code.replace("(", "").replace(")", "")

    new_img = load_img(path+'test\\'+file)
    arr_img = img_to_array(new_img)
    img = arr_img.reshape((1,)+arr_img.shape)
    if i == 0 :
        container = img
        labels = word[label]
    else:
        container = np.vstack([container, img])
        labels = np.vstack([labels, word[label]])

xTest = container
yTest = np_utils.to_categorical(labels,79)

xTrain = xTrain.astype('float32')/255
xTest = xTest.astype('float32')/255

xy = (xTrain, xTest, yTrain, yTest)


