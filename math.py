import os, re, glob, pickle
import cv2
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


dataset_folder_path = './data/extracted_images/'
categories = ["(", ")", "+", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "=", "A",
              "alpha", "b", "beta", "C", "cos", "d", "div", "e", "f",
              "forward_slash", "G", "gamma", "H", "i", "infty", "int", "j", "k", "l",
              "lim", "log", "M", "N", "p", "pi", "q",
              "rightarrow", "sigma", "sin", "sqrt", "sum", "T", "tan", "times", "u", "v", "w", "X", "y",
              "z", "{", "}"]
num_categories = len(categories)
image_w = 45
image_h = 45

X = []
Y = []

for index, category in enumerate(tqdm(categories)):
    label = [0 for i in range(num_categories)]
    label[index] = 1
    image_dir = dataset_folder_path + category + '/'

    for root, dirs, files in os.walk(image_dir):
        for filename in files:
            img = cv2.imread(image_dir+filename)
            img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
            X.append(img/256)
            Y.append(label)

X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y)

xy = (X_train, X_test, Y_train, Y_test)

with open("./img_data.pickle", "wb") as f:
    pickle.dump(xy, f)


