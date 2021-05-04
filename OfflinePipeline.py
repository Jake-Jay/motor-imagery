import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
import scipy.signal
from numpy import linalg
from util.Dataloader import Dataloader


# Load data
dataloader = Dataloader('/BF_MI_05')
data, labels = dataloader.get_data()

#
