from pathlib import Path
import matplotlib.pyplot as plt
from input_adapter import EEGDataAdapter


def get_project_root() -> Path:
    return Path(__file__).parent.parent


class Visualizer:
    def __init__(self, file_name):

        self.data_dir = str(get_project_root()) + "/data" + file_name
        self.eeg_data, self.eeg_labels = self.load_data()

    def load_data(self):
        loader = EEGDataAdapter('', channel_indices=[1, 2, 3, 4, 5, 6, 7, 8], mode='offline',
                                event_dict={'noblink': 0, 'break': 1, 'imagery_handL': 2,
                                            'imagery_handR': 3, 'imagery_foot': 4})

        eeg_data, eeg_labels = loader.load_recorded_eeg_data_file(self.data_dir, file_type='xdf')
        eeg_data = eeg_data * 1e6  # scale for unicorn
        return eeg_data, eeg_labels

    def plot_data(self):
        fig, axs = plt.subplots(9)
        fig.suptitle('EEG Channels')
        for i in range(8):
            axs[i].plot(self.eeg_data[i])
        axs[8].plot(self.eeg_labels)
        fig.show()
        return


my_visualizer = Visualizer('/BF_MI_05')
my_visualizer.plot_data()









