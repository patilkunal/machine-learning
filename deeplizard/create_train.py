import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy


train_labels = []
train_samples = []

# Create our own data here of imaginary clinical trial
for i in range(50):
    # The ~5% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1) # denotes they did experiences side effects

    # The ~5% of older individuals who did NOT experience side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0) # denotes they did NOT experiences side effects

for i in range(1000):
    # The ~95% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0) # denotes they did NOT experiences side effects

    # The ~95% of older individuals who did NOT experience side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1) # denotes they did experiences side effects

# Now process above data

# Make them as numpy array
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
# Shuffle them to make them random
train_labels, train_samples = shuffle(train_labels, train_samples)

# normalize the age data to make them in range of 0 to 1 (as against 13 - 100)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1))

# just print scaled data
# for i in scaled_train_samples:
#    print(i)

model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2,  activation='softmax') # units = 2 since we need two outputs (did or did not experience )
])

model.summary()

# Compile it
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train it
model.fit(x=scaled_train_samples, y=train_labels, batch_size=10, epochs=50, shuffle=True, verbose=2)

