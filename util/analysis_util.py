import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
import matplotlib.transforms as mtransforms


def psd(trials, nchannels, nsamples, fs):
    '''
    Calculates for each trial the Power Spectral Density (PSD).
    
    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEG signal
    
    Returns
    -------
    trial_PSD : 3d-array (channels x PSD x trials)
        the PSD for each trial.  
    freqs : list of floats
        Yhe frequencies for which the PSD was computed (useful for plotting later)
    '''
    
    ntrials = trials.shape[2]
    trials_PSD = np.zeros((nchannels, (nsamples//2) + 1, ntrials)) #changed to 1001

    # Iterate over trials and channels
    for trial in range(ntrials):
        for ch in range(nchannels):
            # Calculate the PSD
            (PSD, freqs) = mlab.psd(trials[ch,:,trial], NFFT=int(nsamples), Fs=fs)
            trials_PSD[ch, :, trial] = PSD.ravel()
                
    return trials_PSD, freqs


def bandpower(trials, fs, band=[8,16]):
    nchannels, nsamples, ntrials = trials.shape
    bandpower = np.zeros((nchannels, ntrials)) 

    # Iterate over trials and channels
    for trial in range(ntrials):
        for ch in range(nchannels):
            # Calculate the PSD
            (PSD, freqs) = mlab.psd(trials[ch,:,trial], NFFT=int(nsamples), Fs=fs)        
            idxs = np.where(np.logical_and(freqs > band[0], freqs < band[1]))
            freqs = freqs[idxs]
            bandpower[ch, trial] = PSD[idxs].mean()
    
    return freqs, bandpower

def plot_bandpower(trials, classes):
    '''
    Plots the bandpower of each channel/component.
    arguments:
        trials - Dictionary containing the trials (log-vars x trials) for 2 classes.
    '''
    plt.figure(figsize=(12,5))

    nchannels = trials[classes[0]].shape[0]

    for i in range(len(classes)):

        x = np.arange(nchannels) + 0.2 * i
        y = np.mean(trials[classes[i]], axis=1)
        plt.bar(x, y, width=0.2)

    plt.xlim(-0.5, nchannels+0.5)

    plt.gca().yaxis.grid(True)
    plt.title('Bandpower of each channel')
    plt.xlabel('channels')
    plt.xticks(x, ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8'])
    plt.ylabel('bandpower')
    plt.legend(classes)

def plot_psd(trials_PSD, freqs, chan_ind, chan_lab=None, maxy=None):
    '''
    Plots PSD data calculated with psd().
    
    Parameters
    ----------
    trials : 3d-array
        The PSD data, as returned by psd()
    freqs : list of floats
        The frequencies for which the PSD is defined, as returned by psd() 
    chan_ind : list of integers
        The indices of the channels to plot
    chan_lab : list of strings
        (optional) List of names for each channel
    maxy : float
        (optional) Limit the y-axis to this value
    '''
    plt.figure(figsize=(12,5))
    
    nchans = len(chan_ind)
    
    # Maximum of 3 plots per row
    nrows = int(np.ceil(nchans / 3))
    ncols = min(3, nchans)
    
    # Enumerate over the channels
    for i,ch in enumerate(chan_ind):
        # Figure out which subplot to draw to
        ax = plt.subplot(nrows,ncols,i+1)
    
        # Plot the PSD for each class
        for cl in trials_PSD.keys():
            plt.plot(freqs, np.mean(trials_PSD[cl][ch,:,:], axis=1), label=cl)
    
        # All plot decoration below...

        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.fill_between(freqs, 0, 1, where=np.logical_and(freqs > 8, freqs < 16),
                facecolor='green', alpha=0.2, transform=trans, label='Alpha Band')

        plt.xlim(1,30)
        
        if maxy != None:
            plt.ylim(0,maxy)
    
        plt.grid()
    
        plt.xlabel('Frequency (Hz)')
        
        if chan_lab == None:
            plt.title('Channel %d' % (ch+1))
        else:
            plt.title(chan_lab[i])

        plt.legend()
        
    plt.tight_layout()

# Calculate the log(var) of the trials
def logvar(trials):
    '''
    Calculate the log-var of each channel.
    
    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEG signal.
        
    Returns
    -------
    logvar - 2d-array (channels x trials)
        For each channel the logvar of the signal
    '''
    return np.log(np.var(trials, axis=1))

def plot_logvar(trials, classes):
    '''
    Plots the log-var of each channel/component.
    arguments:
        trials - Dictionary containing the trials (log-vars x trials) for 2 classes.
    '''
    plt.figure(figsize=(12,5))

    nchannels = trials[classes[0]].shape[0]
    
    x0 = np.arange(nchannels)
    x1 = np.arange(nchannels) + 0.4

    y0 = np.mean(trials[classes[0]], axis=1)
    y1 = np.mean(trials[classes[1]], axis=1)

    plt.bar(x0, y0, width=0.5, color='b')
    plt.bar(x1, y1, width=0.4, color='r')

    plt.xlim(-0.5, nchannels+0.5)

    plt.gca().yaxis.grid(True)
    plt.title('log-var of each channel/component')
    plt.xlabel('channels/components')
    plt.ylabel('log-var')
    plt.legend(classes)

# Scatter plot
def plot_scatter(cl0, cl1, cl_lab):
    plt.figure()
    plt.scatter(cl0[0,:], cl0[-1,:], color='b')
    plt.scatter(cl1[0,:], cl1[-1,:], color='r')
    plt.xlabel('Last component')
    plt.ylabel('First component')
    plt.legend(cl_lab)