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

## ADDTIONAL CODE to validation_set.py
test_labels = []
test_samples = []

# Create our own data here of imaginary clinical trial
for i in range(10):
    # The ~5% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(1) # denotes they did experiences side effects

    # The ~5% of older individuals who did NOT experience side effects
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(0) # denotes they did NOT experiences side effects

for i in range(200):
    # The ~95% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(0) # denotes they did NOT experiences side effects

    # The ~95% of older individuals who did NOT experience side effects
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(1) # denotes they did experiences side effects


# Now process above data

# Make them as numpy array
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
# Shuffle them to make them random
train_labels, train_samples = shuffle(train_labels, train_samples)

test_labels = np.array(test_labels)
test_samples = np.array(test_samples)
# Shuffle them to make them random
test_labels, test_samples = shuffle(test_labels, test_samples)

# normalize the age data to make them in range of 0 to 1 (as against 13 - 100)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1))
scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1, 1))

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

# Train it and split 10% of data as validation set
# even though shuffle is true, validation set is seperated before shuffle, so it may not contain random data as we want
# we will see val_loss & val_accuracy output as against when we did not specify the validation split param
model.fit(x=scaled_train_samples, y=train_labels, validation_split=0.1, batch_size=10, epochs=57, shuffle=True, verbose=2)

# NOW PREDICT using test samples
predictions = model.predict(x=scaled_test_samples, batch_size=10, verbose=0)

for i in predictions:
    print(i)
# outputs in following format
# no side effect, has side effect
# [0.08610389 0.913896  ]
# [0.96067905 0.03932092]


rounded_predictions = np.argmax(predictions, axis=-1)

for i in rounded_predictions:
    print(i)
# rounding is easy to visualize above matrix
# 1 - has side effect
# 0 - no side effect
    