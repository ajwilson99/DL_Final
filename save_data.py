import numpy as np
from gwt import gwt
from scipy.io import loadmat
import time

def save_data():

    signals = loadmat('WiFi_colPre_cfoRem')['colPre_cfoRem']

    # Don't change the parameters below
    M = 386  # <-
    K = 386  # <-
    Fs = 23750000  # <-
    Ndelta = 1  # <-
    WdthG = 0.015  # <-
    plot = False  # <-

    TFRs = np.zeros([signals.shape[0], M, K, signals.shape[2]])

    targets = np.hstack([0*np.ones(signals.shape[0]), np.ones(signals.shape[0]), \
                        2*np.ones(signals.shape[0]), 3*np.ones(signals.shape[0])]).astype(int)

    start_time = time.time()
    for dev in range(0, signals.shape[2]):

        print("Transforming device {} signals...".format(dev + 1))

        for sig in range(0, signals.shape[0]):
            current_signal = signals[sig, :, dev]

            (tfr, _, _) = gwt(current_signal, M, K, Fs, Ndelta, WdthG, plot)
            TFRs[sig, :, :, dev] = tfr

        print("done.\n")

    TFRs = TFRs.reshape(signals.shape[0] * signals.shape[2], M, K, 1)

    end_time = time.time()

    transform_time = end_time - start_time
    print("Time to compute all GT's: {} minutes\n".format(transform_time / 60))

    np.save('tfrs.npy', TFRs)
    np.save('targets.npy', targets)