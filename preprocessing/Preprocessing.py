import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))
    __package__ = "preprocessing"

import util.Dataloader
# from util.Dataloader import Dataloader


class Preprocessing:
    def __init__(self, file_name):

        dataloader = Dataloader(file_name)
        self.eeg_data, self.eeg_labels = dataloader.get_data()


preprocessor = Preprocessing('/BF_MI_05')
