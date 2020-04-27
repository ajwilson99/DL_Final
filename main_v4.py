import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LeakyReLU, Conv2D, MaxPooling2D, SpatialDropout2D, Dense, Flatten, Input, Add
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold

def main():

    gpus = tf.config.experimental.list_physical_devices('GPU')

    # Configuring GPUs
    print("Configuring GPUs...")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Load data and labels
    print("Loading data and target labels...")
    TFRs = np.load('tfrs.npy')
    targets = np.load('targets.npy')
    # y_binary = to_categorical(targets)

    print("Building model...")

    input_layer = Input(shape=(128, 128, 3))
    act = LeakyReLU(alpha=0.1)

    conv_layer_1 = Conv2D(144, (5, 5), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    dropout_layer_1 = SpatialDropout2D(0.5)(conv_layer_1)
    pool_layer_1 = MaxPooling2D((2, 2))(dropout_layer_1)

    conv_layer_2 = Conv2D(96, (3, 3), strides=(1, 1), padding='valid', activation='relu')(pool_layer_1)
    dropout_layer_2 = SpatialDropout2D(0.5)(conv_layer_2)   
    pool_layer_2 = MaxPooling2D((2, 2))(dropout_layer_2)

    conv_layer_3 = Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu')(pool_layer_2)
    dropout_layer_3 = SpatialDropout2D(0.5)(conv_layer_3)
    pool_layer_3 = MaxPooling2D((2, 2))(dropout_layer_3)

    conv_layer_4 = Conv2D(24, (3, 3), strides=(1, 1), padding='valid', activation='relu')(pool_layer_3)
    dropout_layer_4 = SpatialDropout2D(0.5)(conv_layer_4)
    pool_layer_4 = MaxPooling2D((2, 2))(dropout_layer_4)

    flat_layer = Flatten()(pool_layer_4)
    dense_layer = Dense(128, activation='relu')(flat_layer)
    output_layer = Dense(4, activation='softmax')(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()

    comp = keras.optimizers.SGD(learning_rate=0.00005) 
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(TFRs, targets, epochs=1000)
    
if __name__ == "__main__":
    main()
