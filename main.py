""" main.py
 Written by Aaron Wilson & Bohan Li
 COSC 525: Deep Learning, Spring 2020
 Final Project
 April, 2020 """

import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LeakyReLU, Conv2D, MaxPooling2D, SpatialDropout2D, Dense, Flatten, Input, Add
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import datasets
from sklearn.metrics import confusion_matrix as CM

def main():

    gpus = tf.config.experimental.list_physical_devices('GPU')

    # Configuring GPUs
    print("Configuring GPUs...")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Load data and labels
    train_x = np.load('tfrs.npy').reshape(8000, 386, 386, 1)
    train_y = np.load('targets.npy')

    # Develop CNN model
    print("Building model...")

    input_layer = Input(shape=(386, 386, 1))
    act = LeakyReLU(alpha=0.1)

    conv_layer_1 = Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation=act)(input_layer)
    dropout_layer_1 = SpatialDropout2D(0.5)(conv_layer_1)
    pool_layer_1 = MaxPooling2D((2, 2))(dropout_layer_1)

    down_samp_layer_1 = Conv2D(32, (1, 1), strides=(1, 1), padding='same')(pool_layer_1)
    conv_layer_2a = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation=act)(down_samp_layer_1)
    conv_layer_2b = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation=act)(conv_layer_2a)
    skip_layer_1 = Add()([down_samp_layer_1, conv_layer_2b])

    dropout_layer_2 = SpatialDropout2D(0.5)(skip_layer_1)
    pool_layer_2 = MaxPooling2D((2, 2))(dropout_layer_2)

    down_samp_layer_2 = Conv2D(12, (1, 1), strides=(1, 1), padding='same')(pool_layer_2)
    conv_layer_3a = Conv2D(12, (3, 3), strides=(1, 1), padding='same', activation=act)(down_samp_layer_2)
    conv_layer_3b = Conv2D(12, (3, 3), strides=(1, 1), padding='same', activation=act)(conv_layer_3a)
    skip_layer_2 = Add()([down_samp_layer_2, conv_layer_3b])

    dropout_layer_3 = SpatialDropout2D(0.5)(skip_layer_2)
    pool_layer_3 = MaxPooling2D((2, 2))(dropout_layer_3)

    down_samp_layer_3 = Conv2D(8, (1, 1), strides=(1, 1), padding='same')(pool_layer_3)
    conv_layer_4a = Conv2D(8, (3, 3), strides=(1, 1), padding='same', activation=act)(down_samp_layer_3)
    conv_layer_4b = Conv2D(8, (3, 3), strides=(1, 1), padding='same', activation=act)(conv_layer_4a)
    skip_layer_3 = Add()([down_samp_layer_3, conv_layer_4b])

    dropout_layer_4 = SpatialDropout2D(0.5)(skip_layer_3)
    pool_layer_4 = MaxPooling2D((2, 2))(dropout_layer_4)

    flat_layer = Flatten()(pool_layer_4)
    dense_layer = Dense(128, activation=act)(flat_layer)
    output_layer = Dense(4, activation='softmax')(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    hist = model.fit(train_x, train_y, epochs=20, validation_split=0.20)

    np.save('training_accuracy.npy', hist.history['accuracy'])
    np.save('training_loss.npy', hist.history['loss'])
    np.save('validation_accuracy.npy', hist.history['val_accuracy'])
    np.save('validation_loss.npy', hist.history['val_loss'])

if __name__ == "__main__":
    main()
