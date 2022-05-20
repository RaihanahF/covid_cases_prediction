# -*- coding: utf-8 -*-
"""
Created on Fri May 20 12:04:02 2022

@author: Fatin
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import TensorBoard

DATASET_TRAIN_PATH = os.path.join(os.getcwd(), 'cases_malaysia_train.csv')
DATASET_TEST_PATH = os.path.join(os.getcwd(), 'cases_malaysia_test.csv')
LOG_PATH = os.path.join(os.getcwd(),'log')

#%% EDA

# 1: Data Loading
X_train = pd.read_csv(DATASET_TRAIN_PATH)
X_test = pd.read_csv(DATASET_TEST_PATH)

# 2: Data interpretation
X_train.info()
X_train.describe()
print(X_train.isna().sum())
print(X_test.isna().sum())

X_train['cases_new'] = X_train['cases_new']. replace (" ", 0)
X_train['cases_new'] = X_train['cases_new']. replace ("?", 0)
X_test['cases_new'] = X_test['cases_new'].fillna(0)

x_train = X_train['cases_new'].values
x_test = X_test['cases_new'].values

# 3: Data visualization

plt.figure()
#plt.plot(x_train)
plt.show()

# 4: Data cleaning
# 5: Feature Selection
# 6: Data Preprocessing

mms = MinMaxScaler()
x_train_scaled = mms.fit_transform(np.expand_dims(x_train, -1))
x_test_scaled = mms.fit_transform(np.expand_dims(x_test, -1))

x_train = []
y_train = []

# 30 days
for i in range (30,len(x_train_scaled)):
    x_train.append(x_train_scaled[i-30:i,0])
    y_train.append(x_train_scaled[i,0])

# testing dataset
dataset_total = np.concatenate((x_train_scaled, x_test_scaled), axis=0)
data = dataset_total[-50:]

x_test = []
y_test = []

for i in range (30,50):
    x_test.append(data[i-30:i,0])
    y_test.append(data[i,0])
    
x_train = np.array(x_train)
x_train = np.expand_dims(x_train, axis=-1)

y_train = np.array(y_train)

x_test = np.array(x_test)
x_test = np.expand_dims(x_test, axis=-1)

y_test = np.array(y_test)

#%% Model Creation

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential()
model.add(LSTM(64, activation='tanh',
               return_sequences=True,
               input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(1, activation='relu'))

model.summary()

tf.keras.utils.plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

log_dir = os.path.join(LOG_PATH,
                       datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Step 5) Model training
model.compile(optimizer='adam', loss='mse', metrics='mse')

hist = model.fit(x_train, y_train, epochs=50, callbacks=tensorboard_callback)

print(hist.history.keys)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(hist.history['loss'])
plt.xlabel('epochs')
plt.ylabel('training loss')
plt.show()

#%% Model Deployment

# data prediction
predicted = []

for i in x_test:
    predicted.append(model.predict(np.expand_dims(i,axis=0)))

plt.figure()
plt.plot(np.array(predicted).reshape(20,1), color='r')
plt.plot(y_test, color='b')
plt.legend(['predicted', 'actual'])
plt.show()

inversed_y_true = mms.inverse_transform(np.expand_dims(y_test, axis=-1))
inversed_y_pred = mms.inverse_transform(np.array(predicted).reshape(20,1))

plt.figure()
plt.plot(inversed_y_pred, color='r')
plt.plot(inversed_y_true, color='b')
plt.legend(['predicted', 'actual'])
plt.show()

#%% Performance Evaluation

from sklearn.metrics import mean_absolute_error

y_true = y_test
y_predicted = np.array(predicted).reshape(20,1)

print(mean_absolute_error(y_test, y_predicted)/sum(y_predicted)*100)


