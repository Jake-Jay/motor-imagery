
# %%
import numpy as np
import mne
import matplotlib.pyplot as plt
from pathlib import Path

from input_adapter import EEGDataAdapter
import pdb


loader = EEGDataAdapter('', 
    channel_indices=[1, 2, 3, 4, 5, 6, 7, 8], 
    mode='offline',
    event_dict={
        'noblink': 0, 
        'break': 1, 
        'imagery_handL': 2, 
        'imagery_handR': 3, 
        'imagery_foot': 4
    }
)

DATA_FILE_NAME = '../../data/BF_MI_05'

# Load data
eeg_data, eeg_labels = loader.load_recorded_eeg_data_file(
    DATA_FILE_NAME, file_type='xdf'
)
eeg_data = eeg_data * 1e6  # scale for unicorn
reref_eeg_data = eeg_data - np.mean(eeg_data, axis=0)

# Plot data
fig, axs = plt.subplots(9)
fig.suptitle('EEG Channels')
for i in range(8):
    axs[i].plot(reref_eeg_data[i])
axs[8].plot(eeg_labels)

# Create MNE array from numpy data
print('To MNE...')
ch_names = ['EEG Fz', 'EEG C3', 'EEG Cz', 'EEG C4', 'EEG Pz', 'EEG PO7', 'EEG Oz', 'EEG PO8']
ch_types = ["eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg"]
info = mne.create_info(
    ch_names=ch_names,
    sfreq=250, 
    ch_types=ch_types
)
raw = mne.io.RawArray(reref_eeg_data, info)

# Filter using MNE
raw.filter(7., 30., fir_design='firwin')
raw.notch_filter(50, filter_length='auto', fir_design='firwin')


# Plot data
fig, axs = plt.subplots(9)
fig.suptitle('EEG Channels')
for i in range(8):
    axs[i].plot(raw._data[i])
axs[8].plot(eeg_labels)

plt.show()

raw.plot(duration=5)

plt.show()
