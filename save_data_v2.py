import numpy as np
from gwt import gwt
from scipy.io import loadmat
import time
from matplotlib import cm
from scipy.misc import imresize

def save_data():

    signals = loadmat('WiFi_colPre_cfoRem')['colPre_cfoRem']

    # Don't change the parameters below
    M = 386  # <-
    K = 386  # <-
    Fs = 23750000  # <-
    Ndelta = 1  # <-
    WdthG = 0.015  # <-
    plot = False  # <-

    TFRs = np.zeros([signals.shape[0], 100, 100, signals.shape[2]])

    targets = np.hstack([0*np.ones(signals.shape[0]), np.ones(signals.shape[0]), \
                        2*np.ones(signals.shape[0]), 3*np.ones(signals.shape[0])]).astype(int)

    start_time = time.time()
    for dev in range(0, signals.shape[2]):

        print("Transforming device {} signals...".format(dev + 1))

        for sig in range(0, signals.shape[0]):
            current_signal = signals[sig, :, dev]

            (tfr, _, _) = gwt(current_signal, M, K, Fs, Ndelta, WdthG, plot)
            # tfr_rgb = cm.jet(tfr)[:, :, 0:3]

            TFRs[sig, :, :, dev] = np.vstack([tfr[0:50, 250:350], tfr[0:50, 0:100]])

        print("done.\n")

    TFRs = TFRs.reshape(signals.shape[0] * signals.shape[2], 100, 100)
   
    # Standardize images before saving
    print('Standardizing Images...')
    TFRs -= np.mean(TFRs)

    TFRs /= np.std(TFRs)

    end_time = time.time()

    transform_time = end_time - start_time
    print("Time to compute all GT's: {} minutes\n".format(transform_time / 60))

    print('Saving images...')
    np.save('tfrs.npy', TFRs)
    np.save('targets.npy', targets)


if __name__ == "__main__":
    save_data()
