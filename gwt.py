import numpy as np, time
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.signal as signal
from scipy.io import loadmat
from scipy.fftpack import fft

""" Input parameters for wifi signals:

    M: 386
    Nfft: 386
    Fs: from WiFi_colPre_cfoRem['Tsamp']
    Ndelta: 1
    WdthG: 0.015
    plot: True
    """

def ind2sub(array_shape, ind):
    rows = (ind.astype('int') / array_shape[1])
    cols = (ind.astype('int') % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
    return (rows, cols)

# The code below was taken from gwt_v5.py written by Dr. Donald Reising, University of Tennessee at Chattanooga


def gwt(SigIn, M, Nfft, Fs, Ndelta, WdthG, plot):

    if len(SigIn.shape) == 1:
       Ns = len(SigIn)
       m = 1 

    tau = np.linspace(-M/2, M/2 - 1, num=M)
    Ntshfts = len(tau)

    ## Create Gaussian window used to compute GT
    E1 = -np.pi / (WdthG*Ns**2)
    E2 = ((np.linspace(1, Ns, Ns))-0.5*(Ns-1))**2
    winGT = np.exp(E1*E2)

    ## Normalize for sum(|winGT|^2) = 1
    NrmFct = sum(winGT**2)
    winGT = winGT / np.sqrt(NrmFct)

    ## Make the window a matrix
    winGT = np.array([winGT, ]*Ntshfts)

    x = SigIn
    x = np.array([x, ]*Ntshfts)
    
    (c, r) = np.meshgrid(np.linspace(1, Ntshfts, Ntshfts), \
        np.linspace(1, Ns, Ns))

    c = np.mod(c + (tau.reshape(M, 1) - 1), Ns)+1

    xShftd = x[r.astype(int) - 1, c.astype(int) - 1]
    
    gfft = fft(xShftd*winGT, Nfft, axis=1)

    Tscale = np.linspace(0, (Ntshfts-1)*(1/Fs), num=Ntshfts)  # Time scale
    dF = Fs/(2*Nfft);  # Freq Plot Step Size
    Fscale = np.linspace(0, (Nfft-1)*dF, num=Nfft)

    # tfr_mag = 10*np.log10((np.abs(gfft) - np.min(np.min(np.abs(gfft))))
    #          / (np.max(np.max(np.abs(gfft))) - np.min(np.min(np.abs(gfft)))))

    tfr_mag = (np.abs(gfft) - np.min(np.min(np.abs(gfft)))) / (np.max(np.max(np.abs(gfft))) - np.min(np.min(np.abs(gfft))))

    # dB_floor = np.max(np.max(tfr_mag)) - 30
    #
    # ri, ci = np.where(tfr_mag <= dB_floor)
    #
    # tfr_mag[ri, ci] = dB_floor

    # if plot:
    #
    #
    #     # Remove "nasty" exponentials from plotting axii
    #     t_pow = np.log10(np.max(Tscale))
    #     f_pow = np.log10(np.max(Fscale))
    #
    #     # Redo the time axis
    #     if (t_pow >= -3) & (t_pow < 0):
    #         Tscale = Tscale*1e3
    #         time_label = 'Time (ms)'
    #     elif (t_pow >= -6) & (t_pow < -3):
    #         Tscale = Tscale*1e6
    #         time_label = 'Time (microseconds)'
    #     elif (t_pow >= -9) & (t_pow < -6):
    #         Tscale = Tscale*1e9
    #         time_label = 'Time (ns)'
    #     else:
    #         time_label = 'Time (seconds)'
    #
    #     # Redo the frequency axis
    #     if (f_pow >= 3) & (f_pow < 6):
    #         Fscale = Fscale/1e3
    #         freq_label = 'Frequency (kHz)'
    #     elif (f_pow >= 6) & (f_pow < 9):
    #         Fscale = Fscale/1e6
    #         freq_label = 'Frequency (MHz)'
    #     elif (f_pow >=9):
    #         Fscale = Fscale/1e9
    #         freq_label = 'Frequency (GHz)'
    #     else:
    #         freq_label = 'Frequency (Hz)'
    #
    #     # Plotting
    #     cmap = plt.get_cmap('jet')
    #     plt.pcolormesh(Tscale, Fscale, tfr_mag.T, cmap=cmap)
    #     plt.colorbar
    #     plt.xlabel(time_label)
    #     plt.ylabel(freq_label)
    #     plt.show()

    return (tfr_mag, Fscale, Tscale)



## Comment out everything below after testing is done


def main():

    test_signal = loadmat('WiFi_colPre_cfoRem')['colPre_cfoRem'][0, :, 0]

    # plt.plot(test_signal)
    # plt.grid()
    # plt.title('Device 1, Signal 1 WiFi Preamble')

    # Don't change the parameters below
    M = 386  # <-
    K = 386  # <-
    Fs = 23750000  # <-
    Ndelta = 1  # <-
    WdthG = 0.015  # <-
    plot = False  # <-

    start_time = time.time()
    (tfr, f, t) = gwt(test_signal, M, K, Fs, Ndelta, WdthG, plot)
    end_time = time.time()

    duration = end_time - start_time
    print("GWT took roughly {} seconds".format(duration))

    print('GWT of all signals would take approximately {} minutes'.format(8000*duration/60))
    a = 1






if __name__ == "__main__":
    main()