import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from gwt import gwt
from scipy.io import loadmat


def main():

    # Load data and labels
    start_time = time.time()
    TFRs = np.load('tfrs.npy')
    targets = np.load('targets.npy')
    end_time = time.time()
    load_duration = end_time - start_time
    print('Took {} seconds to load data and labels'.format(load_duration))

    y_binary = to_categorical(targets)

    # Develop CNN model
    input_shape = (TFRs.shape[1], TFRs.shape[2], 1)

    model = Sequential()
    model.add(Conv2D(96, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(48, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(24, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.summary()

    comp = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=comp, metrics=['accuracy'])

    start_time = time.time()
    model.fit(TFRs, y_binary, batch_size=100, epochs=5)
    end_time = time.time()
    train_time = end_time - start_time
    print("Time to train model: {} minutes.\n".format(train_time / 60))


if __name__ == "__main__":
    main()
