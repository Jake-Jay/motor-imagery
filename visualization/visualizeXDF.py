from pathlib import Path
import os
import pyxdf
import numpy as np
import matplotlib.pyplot as plt
from data.input_adapter import EEGDataAdapter
from matplotlib import mlab
import scipy.signal
from numpy import linalg
import matplotlib.pyplot as plt


def get_project_root() -> Path:
    return Path(__file__).parent.parent


class Visualizer:
    def __init__(self, file_name):

        self.data_dir = str(get_project_root()) + "/data" + file_name

    def load_data(self):
        loader = EEGDataAdapter('', channel_indices=[1, 2, 3, 4, 5, 6, 7, 8], mode='offline',
                                event_dict={'noblink': 0, 'break': 1, 'imagery_handL': 2,
                                            'imagery_handR': 3, 'imagery_foot': 4})

        eeg_data, eeg_labels = loader.load_recorded_eeg_data_file(self.data_dir, file_type='xdf')
        eeg_data = eeg_data * 1e6  # scale for unicorn
        return eeg_data, eeg_labels

    @staticmethod
    def plot_data():
        fig, axs = plt.subplots(9)
        fig.suptitle('EEG Channels')
        for i in range(8):
            axs[i].plot(eeg_data[i])
        axs[8].plot(eeg_labels)
        fig.show()


my_visualizer = Visualizer('/BF_MI_05')
eeg_data, eeg_labels = my_visualizer.load_data()
my_visualizer.plot_data()








