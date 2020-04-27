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

    # Develop CNN model with skip layers
    print("Building model...")
    input_layer = Input(shape=(128, 128, 3))
    act = LeakyReLU(alpha=1.5)

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

 #   conv_layer_1 = Conv2D(24, (3, 3), strides=(1, 1), padding='valid', activation=act, kernel_initializer='normal')(input_layer)

    #pool_layer_1 = MaxPooling2D((2, 2))(conv_layer_1)
 #   down_samp_layer_1 = Conv2D(36, (1, 1), strides=(1, 1), padding='valid', activation=act, kernel_initializer='normal')(conv_layer_1)
 #   conv_layer_2a = Conv2D(36, (3, 3), strides=(1, 1), padding='same', activation=act, kernel_initializer='normal')(down_samp_layer_1)
 #   conv_layer_2b = Conv2D(36, (3, 3), strides=(1, 1), padding='same', activation=act, kernel_initializer='normal')(conv_layer_2a)
 #   conv_layer_3 = Add()([down_samp_layer_1, conv_layer_2b])  # Skip connection

 #   pool_layer_2 = MaxPooling2D((2, 2))(conv_layer_3)
 #   down_samp_layer_2 = Conv2D(48, (1, 1), strides=(1, 1), padding='valid', activation=act, kernel_initializer='normal')(pool_layer_2)
 #   conv_layer_4a = Conv2D(48, (3, 3), strides=(1, 1), padding='same', activation=act, kernel_initializer='normal')(down_samp_layer_2)
 #   conv_layer_4b = Conv2D(48, (3, 3), strides=(1, 1), padding='same', activation=act, kernel_initializer='normal')(conv_layer_4a)
 #   conv_layer_5 = Add()([down_samp_layer_2, conv_layer_4b])

 #   pool_layer_3 = MaxPooling2D((2, 2))(conv_layer_5)
 #   down_samp_layer_3 = Conv2D(60, (1, 1), strides=(1, 1), padding='valid', activation=act, kernel_initializer='normal')(pool_layer_3)
 #   conv_layer_6a = Conv2D(60, (3, 3), strides=(1, 1), padding='same', activation=act, kernel_initializer='normal')(down_samp_layer_3)
 #   conv_layer_6b = Conv2D(60, (3, 3), strides=(1, 1), padding='same', activation=act, kernel_initializer='normal')(conv_layer_6a)
 #   conv_layer_7 = Add()([down_samp_layer_3, conv_layer_6b])

 #   pool_layer_4 = MaxPooling2D((2, 2))(conv_layer_7)
 #   down_samp_layer_4 = Conv2D(72, (1, 1), strides=(1, 1), padding='valid', activation=act, kernel_initializer='normal')(pool_layer_4)
 #   conv_layer_8a = Conv2D(72, (3, 3), strides=(1, 1), padding='same', activation=act, kernel_initializer='normal')(down_samp_layer_4)
 #   conv_layer_8b = Conv2D(72, (3, 3), strides=(1, 1), padding='same', activation=act, kernel_initializer='normal')(conv_layer_8a)
 #   conv_layer_9 = Add()([down_samp_layer_4, conv_layer_8b])

 #   pool_layer_5 = MaxPooling2D((2, 2))(conv_layer_9)
 #   down_samp_layer_5 = Conv2D(96, (1, 1), strides=(1, 1), padding='valid', activation=act, kernel_initializer='normal')(pool_layer_5)
 #   conv_layer_10a = Conv2D(96, (3, 3), strides=(1, 1), padding='same', activation=act, kernel_initializer='normal')(down_samp_layer_5)
 #   conv_layer_10b = Conv2D(96, (3, 3), strides=(1, 1), padding='same', activation=act, kernel_initializer='normal')(conv_layer_10a)
 #   conv_layer_11 = Add()([down_samp_layer_5, conv_layer_10b])
 
 #   pool_layer_6 = MaxPooling2D((2, 2))(conv_layer_11)
 #   down_samp_layer_6 = Conv2D(108, (1, 1), strides=(1, 1), padding='valid', activation=act, kernel_initializer='normal')(pool_layer_6)
 #   conv_layer_12a = Conv2D(108, (3, 3), strides=(1, 1), padding='same', activation=act, kernel_initializer='normal')(down_samp_layer_6)
 #   conv_layer_12b = Conv2D(108, (3, 3), strides=(1, 1), padding='same', activation=act, kernel_initializer='normal')(conv_layer_12a)
 #   conv_layer_13 = Add()([down_samp_layer_6, conv_layer_12b])

 #   pool_layer_7 = MaxPooling2D((4, 4))(conv_layer_13)
 #   flat_layer = Flatten()(pool_layer_7)
 #   dense_layer_1 = Dense(100, activation=act)(flat_layer)
 #   dense_layer_2 = Dense(10, activation=act)(dense_layer_1)
 #   dense_layer_3 = Dense(4, activation='softmax')(dense_layer_2)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    scores = []
    count = 1
    for train, test in kfold.split(TFRs, targets):

        print("Fitting model on k fold \# {}...".format(count))
        comp = keras.optimizers.SGD(learning_rate=0.00005)   
        model.compile(loss='sparse_categorical_crossentropy', optimizer=comp, metrics=['accuracy'])
        model.fit(TFRs[train], targets[train], epochs=100)
	score_k = model.evaluate(TFRs[test], targets[test])
        print('Kfold {} Accuracy: {}%'.format(count, score_k[1]*100))
	scores.append(score_k[1]*100)
	count += 1

    
if __name__ == "__main__":
    main()
