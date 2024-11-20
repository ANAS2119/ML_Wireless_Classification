import matplotlib.pyplot as plt
import numpy as np
import pickle

with open('RML2016.10a_dict.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')  # Use 'latin1' encoding to avoid Unicode errors

with open('data_details.txt', 'w') as f:
    for key, signals in data.items():
        mod_type, snr = key
        print(f"Modulation Type: {mod_type}, SNR: {snr}, Shape: {signals.shape}", file=f)


# Function to plot all highest SNR signals in one figure with separate subplots
def plot_all_highest_snr_signals(data):
    # Dictionary to store the highest SNR signal for each modulation type
    highest_snr_signals = {}

    # Find the highest SNR signal for each modulation type
    for key, signals in data.items():
        mod_type, snr = key
        if (
            mod_type not in highest_snr_signals
            or snr > highest_snr_signals[mod_type][1]
        ):
            highest_snr_signals[mod_type] = (
                signals[0],
                snr,
            )  # Store the signal and its SNR
            
     # Create a figure and add subplots for each modulation type
    num_modulations = len(highest_snr_signals)
    fig, axs = plt.subplots(num_modulations, 2, figsize=(18, 4 * num_modulations))
    
     # Plot each modulation type's highest SNR signal
    for i, (mod_type, (signal, snr)) in enumerate(highest_snr_signals.items()):
        real_part = signal[0]  # I-component (real)
        imag_part = signal[1]  # Q-component (imaginary)

        # Plot real (I) component
        axs[i, 0].plot(real_part)
        axs[i, 0].set_title(f"Modulation: {mod_type}, SNR: {snr} dB - Real (I)")
        axs[i, 0].set_xlabel("Sample Index")
        axs[i, 0].set_ylabel("Amplitude")

        # Plot imaginary (Q) component
        axs[i, 1].plot(imag_part)
        axs[i, 1].set_title(f"Modulation: {mod_type}, SNR: {snr} dB - Imaginary (Q)")
        axs[i, 1].set_xlabel("Sample Index")
        axs[i, 1].set_ylabel("Amplitude")
    
    plt.tight_layout()
    plt.show()

    return num_modulations

plot_all_highest_snr_signals(data)