# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import pickle
from helper import seprateImagesInClasses, creatELADataSet
import shutil
import glob

#%matplotlib inline

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

sns.set(style='white', context='notebook', palette='deep')

from PIL import Image
import os
from pylab import *
import re
from PIL import Image, ImageChops, ImageEnhance

flag = 'linux'

if 'colab' == flag:
  root_dir = '/content/drive/My Drive/Colab_Notebooks'
elif 'linux' == flag:
  root_dir = os.path.abspath("/home/g1g/Desktop/splice-detection")
else:
  root_dir = os.path.abspath(r"d:/home/g1g/Desktop/splice-detection")

#root_dir = os.path.abspath("/home/g1g/Desktop/splice-detection")

baseDataSetPath = os.path.join(root_dir, 'dataset')
dataSetPath = os.path.join(root_dir, 'dataset2')


manipulated = 'manipulated'
non_manipulated = 'non_manipulated'


#X, Y, Xp, Yp = creatELADataSet(dataSetPath)

# save pickle
#with open('ELAdataset.pickle', 'wb') as f:
#    pickle.dump((X, Y, Xp, Yp), f)

# load pickl()

with open('ELAdataset.pickle', 'rb') as f:
    X, Y, Xp, Yp = pickle.load(f)


X_train, X_val, Y_train, Y_val = train_test_split(Xp, Yp, test_size = 0.2, random_state=5)

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'valid', 
                 activation ='relu', input_shape = (128,128,3)))
print("Input: ", model.input_shape)
print("Output: ", model.output_shape)

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'valid', 
                 activation ='relu'))
print("Input: ", model.input_shape)
print("Output: ", model.output_shape)

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))
print("Input: ", model.input_shape)
print("Output: ", model.output_shape)

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation = "softmax"))
  
model.summary()

#optimizer = RMSprop(lr=0.0005, rho=0.9, epsilon=1e-08, decay=0.0)
optimizer = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

#early_stopping = EarlyStopping(monitor='val_acc',
#                              min_delta=0,
#                              patience=2,
#                              verbose=0, mode='auto')

epochs = 50
batch_size = 100
#callbacks=[early_stopping]
history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, 
          validation_data = (X_val, Y_val), verbose = 2, shuffle=True)

# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


              