import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from util.input_adapter import EEGDataAdapter


def get_project_root() -> Path:
    return Path(__file__).parent.parent


class Dataloader:
    def __init__(self, file_names):

        data_dir = str(get_project_root()) + "/data" + file_names[0]
        self.eeg_data, self.eeg_labels = self.load_data(data_dir)

        if len(file_names) > 1:
            for file_name in file_names[1:]:
                data_dir = str(get_project_root()) + "/data" + file_name
                print(data_dir)
                eeg_data, eeg_labels = self.load_data(data_dir)

                self.eeg_data = np.concatenate([self.eeg_data, eeg_data], axis=1) 
                self.eeg_labels = np.concatenate([self.eeg_labels, eeg_labels]) 

    def load_data(self, data_dir):
        loader = EEGDataAdapter('Graz', channel_indices=[1, 2, 3, 4, 5, 6, 7, 8], mode='offline',
                                event_dict={'noblink': 0,
                                            'break': 1,
                                            'imagery_handL': 2,
                                            'imagery_handR': 3,
                                            }
                                )

        eeg_data, eeg_labels = loader.load_recorded_eeg_data_file(
            data_dir, 
            file_type='xdf',
            recording_paradigm='Graz'
        )
        eeg_data = eeg_data * 1e6  # Scale for unicorn
        return eeg_data, eeg_labels

    def get_data(self):
        return self.eeg_data, self.eeg_labels

    def plot_data(self):
        fig, axs = plt.subplots(9)
        fig.suptitle('EEG Channels')
        for i in range(8):
            axs[i].plot(self.eeg_data[i])
        axs[8].plot(self.eeg_labels)
        fig.show()
        return










