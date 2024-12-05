import pickle
import numpy as np
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch
import pandas as pd


def features_extraction_(data):
    
    print((data.keys()))
    array_of_data = []

    #df prepration
    columnlabels = ["signal_type", "snr", "magnitude_mean", "magnitude_std", "magnitude_skew", "magnitude_kurtosis", "phase_mean", "phase_std", "phase_skew", "phase_kurtosis", "spectral_entropy", "peak_frequency", "average_power"]
    for key in data.keys(): # key dict has 220 len.
        signal_type = key[0]
        # print(signal_type)
        if signal_type in ["BPSK","QPSK","QAM16","WBFM","GFSK"]:
            snr = key[1]
            if int(snr) =< 100: # >= 0 for only non-ngative SNR
                samples = data[key]
                # calculate the magnitude anf phase for each signal (sample)
                for i in range(len(samples)): # each i represent a sample or signal, total 1000 signals
                    i_samples = samples[i][0] # each sample has 128 bits for real part
                    q_samples = samples[i][1] # each sample has 128 bits for imaginery part
                    magnitude = np.sqrt(i_samples**2 + q_samples**2) #array with len 128
                    phase = np.arctan2(q_samples, i_samples) # array with len 128

                    # Calculate statistical features for magnitude
                    magnitude_mean = np.mean(magnitude)
                    magnitude_std = np.std(magnitude)
                    magnitude_skew = skew(magnitude)
                    magnitude_kurtosis = kurtosis(magnitude)

                    # Calculate statistical features for phase
                    phase_mean = np.mean(phase)
                    phase_std = np.std(phase)
                    phase_skew = skew(phase)
                    phase_kurtosis = kurtosis(phase)

                    # calculate the spectral entropy
                    frequencies, psd = welch(magnitude, nperseg=128)
                    psd_norm = psd / np.sum(psd)
                    spectral_entropy = entropy(psd_norm)

                    peak_frequency = frequencies[np.argmax(psd)]

                    average_power = np.mean(psd)

                    # fill the data array ( array_of_data)
                    array_of_data.append([signal_type, snr, magnitude_mean, magnitude_std, magnitude_skew, magnitude_kurtosis, phase_mean, phase_std, phase_skew, phase_kurtosis, spectral_entropy, peak_frequency, average_power])
    
    df = pd.DataFrame(array_of_data, columns = columnlabels) # define the complete dataset with rows (samples) and columns ( features)

    # create a csv file 
    df.to_csv("extracted_features_5_types_all_snr.csv", index=False)
    
    return df
