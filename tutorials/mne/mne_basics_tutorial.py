
# %%
import numpy as np
import mne
import pyxdf

from pathlib import Path

DATA_FILE = '../../data/BF_MI_05.xdf'
# %%
streams, header = pyxdf.load_xdf(DATA_FILE)

# %%
openvibe_stream = streams[0]
unicorn_stream = streams[1]

sfreq = float(unicorn_stream["info"]["nominal_srate"][0])
event_labels = openvibe_stream["time_series"]
data = unicorn_stream["time_series"][:, :8].T

reref_data = data - np.mean(data, axis=0)

# %%
info = mne.create_info(8, sfreq, ch_types=["eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg"])
# %%

raw = mne.io.RawArray(data, info)
raw.filter(7., 30., fir_design='firwin')
# %%
raw.plot( duration=1, start=2,)
# %%
