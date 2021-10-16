from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.layers import Flatten, Convolution2D, MaxPooling2D
from keras.models import load_model
import cv2
import pickle

with open('./img_data.pickle', 'rb') as f:
    X_train, X_test, Y_train, Y_test = pickle.load(f)

categories = ["(", ")", "+", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "=", "A",
              "alpha", "b", "beta", "C", "cos", "d", "div", "e", "f",
              "forward_slash", "G", "gamma", "H", "i", "infty", "int", "j", "k", "l",
              "lim", "log", "M", "N", "p", "pi", "q",
              "rightarrow", "sigma", "sin", "sqrt", "sum", "T", "tan", "times", "u", "v", "w", "X", "y",
              "z", "{", "}"]
num_categories = len(categories)

model = Sequential()
model.add(Convolution2D(16, 3, 3, border_mode='same', activation='relu', input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_categories, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=32, nb_epoch=20)

model.save('Gersang.h5')