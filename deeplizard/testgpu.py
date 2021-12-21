import numpy as np
import tensorflow as tf
from tensorflow import keras

phy_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs avail: ", len(phy_devices))
# tf.config.experimental.set_memory_growth(phy_devices[0], True)
