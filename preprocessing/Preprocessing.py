import numpy as np
import matplotlib.pyplot as plt
from util.Dataloader import Dataloader


class Preprocessing:
    def __init__(self, file_name):

        dataloader = Dataloader(file_name)
        self.eeg_data, self.eeg_labels = dataloader.get_data()


preprocessor = Preprocessing('/BF_MI_05')
