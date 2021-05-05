import numpy as np
import matplotlib.pyplot as plt
# from Recording.Utilities.input_adapter import EEGDataAdapter
from matplotlib import mlab
import scipy.signal
from numpy import linalg


class Preprocessing:
    def __init__(self, eeg_data, eeg_labels):
        self.eeg_data, self.eeg_labels = eeg_data, eeg_labels
        self.sample_rate = 250
        self.nchannels, self.nsamples = self.eeg_data.shape
        self.channel_names = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']
        #  classes we want to separate and their respective codings from above
        self.cl1 = 'left'
        self.cl2 = 'right'
        self.coding1 = 2  # make sure you get the right indices here
        self.coding2 = 3

        self.win = np.arange(int(0.5 * self.sample_rate), int(2.5 * self.sample_rate))
        self.nsamples_per_trial = len(self.win)

    def get_onsets(self):
        onsets = []
        event_codes = []
        for i in range(len(self.eeg_labels) - 1):
            if self.eeg_labels[i] - self.eeg_labels[i + 1] != 0:
                onsets.append(i + 1)
                event_codes.append(self.eeg_labels[i + 1])
        return np.asarray(onsets), np.asarray(event_codes)

    def define_trials(self, onsets, event_codes):
        trials = {}
        for cl, event in zip([self.cl1, self.cl2], [self.coding1, self.coding2]):  # handl and handR
            class_onsets = onsets[event_codes == event]

            # Allocate memory for trials of class
            trials[cl] = np.zeros((self.nchannels, self.nsamples_per_trial, len(class_onsets)))

            for i, onset in enumerate(class_onsets):
                trials[cl][:, :, i] = self.eeg_data[:, self.win + onset]
        print('Shape of trials[left]:', trials[self.cl1].shape)
        print('Shape of trials[right]:', trials[self.cl2].shape)
        return trials

    def psd(self, trials):
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
            The frequencies for which the PSD was computed (useful for plotting later)
        '''
        ntrials = trials.shape[2]
        trials_PSD = np.zeros((self.nchannels, self.sample_rate + 1, ntrials))
        # Iterate over trials and channels
        for trial in range(ntrials):
            for ch in range(self.nchannels):
                # Calculate the PSD
                (PSD, freqs) = mlab.psd(trials[ch, :, trial], NFFT=int(self.nsamples_per_trial), Fs=self.sample_rate)
                trials_PSD[ch, :, trial] = PSD.ravel()

        return trials_PSD, freqs

    def plot_psd(self, trials, trials_PSD, freqs, chan_lab=None, maxy=None, chan_ind = None):
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

        # chan_ind = [self.channel_names.index(ch) for ch in ['C3', 'Cz', 'C4']]

        plt.figure(figsize=(12, 5))
        nchans = len(chan_ind)

        # Maximum of 3 plots per row
        nrows = int(np.ceil(nchans / 3))
        ncols = min(3, nchans)
        # print(nrows)
        # print(ncols)

        # Enumerate over the channels
        for i, ch in enumerate(chan_ind):
            # Figure out which subplot to draw to
            plt.subplot(nrows, ncols, i + 1)

            # Plot the PSD for each class
            for cl in trials.keys():
                plt.plot(freqs, np.mean(trials_PSD[cl][ch, :, :], axis=1), label=cl)

            # All plot decoration below...

            plt.xlim(1, 30)

            if maxy != None:
                plt.ylim(0, maxy)

            plt.grid()

            plt.xlabel('Frequency (Hz)')

            if chan_lab == None:
                plt.title('Channel %d' % (ch + 1))
            else:
                plt.title(chan_lab[i])

            plt.legend()
            # plt.tight_layout()
            plt.show()

    def bandpass(self, trials, lo, hi, sample_rate):
        '''
        Designs and applies a bandpass filter to the signal.

        Parameters
        ----------
        trials : 3d-array (channels x samples x trials)
            The EEGsignal
        lo : float
            Lower frequency bound (in Hz)
        hi : float
            Upper frequency bound (in Hz)
        sample_rate : float
            Sample rate of the signal (in Hz)

        Returns
        -------
        '''
        a, b = scipy.signal.iirfilter(6, [lo / (sample_rate / 2.0), hi / (sample_rate / 2.0)])

        # Applying the filter to each trial
        ntrials = trials.shape[2]
        trials_filt = np.zeros((self.nchannels, self.nsamples_per_trial, ntrials))
        for i in range(ntrials):
            trials_filt[:, :, i] = scipy.signal.filtfilt(a, b, trials[:, :, i], axis=1)

        return trials_filt

    def logvar(self, trials):
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

    def plot_logvar(self, trials):
        '''
        Plots the log-var of each channel/component.
        arguments:
            trials - Dictionary containing the trials (log-vars x trials) for 2 classes.
        '''
        plt.figure(figsize=(12, 5))

        x0 = np.arange(self.nchannels)
        x1 = np.arange(self.nchannels) + 0.4

        y0 = np.mean(trials[self.cl1], axis=1)
        y1 = np.mean(trials[self.cl2], axis=1)

        plt.bar(x0, y0, width=0.5, color='b', label=self.cl1)
        plt.bar(x1, y1, width=0.4, color='r', label=self.cl2)

        plt.xlim(-0.5, self.nchannels + 0.5)

        plt.gca().yaxis.grid(True)
        plt.title('log-var of each channel/component')
        plt.xlabel('channels/components')
        plt.ylabel('log-var')
        plt.legend()
        plt.show()

    def cov(self, trials):
        ''' Calculate the covariance for each trial and return their average '''
        ntrials = trials.shape[2]
        covs = [trials[:, :, i].dot(trials[:, :, i].T) / self.nsamples for i in range(ntrials)]
        return np.mean(covs, axis=0)

    def whitening(self, sigma):
        ''' Calculate a whitening matrix for covariance matrix sigma. '''
        U, l, _ = linalg.svd(sigma)
        return U.dot(np.diag(l ** -0.5))

    def csp(self, trials_r, trials_f):
        '''
        Calculate the CSP transformation matrix W.
        arguments:
            trials_r - Array (channels x samples x trials) containing right hand movement trials
            trials_f - Array (channels x samples x trials) containing foot movement trials
        returns:
            Mixing matrix W
        '''
        cov_r = self.cov(trials_r)
        cov_f = self.cov(trials_f)
        P = self.whitening(cov_r + cov_f)
        B, _, _ = linalg.svd(P.T.dot(cov_f).dot(P))
        W = P.dot(B)
        return W

    def apply_mix(self, W, trials):
        ''' Apply a mixing matrix to each trial (basically multiply W with the EEG signal matrix)'''
        ntrials = trials.shape[2]
        trials_csp = np.zeros((self.nchannels, self.nsamples_per_trial, ntrials))
        for i in range(ntrials):
            trials_csp[:, :, i] = W.T.dot(trials[:, :, i])
        return trials_csp

    def plot_scatter(self, left, right):
        plt.figure()
        plt.scatter(left[0, :], left[-1, :], color='b', label=self.cl1)
        plt.scatter(right[0, :], right[-1, :], color='r', label=self.cl2)
        plt.xlabel('Last component')
        plt.ylabel('First component')
        plt.legend()
        # plt.show()

    def train_lda(self, class1, class2):
        '''
        Trains the LDA algorithm.
        arguments:
            class1 - An array (observations x features) for class 1
            class2 - An array (observations x features) for class 2
        returns:
            The projection matrix W
            The offset b
        '''
        nclasses = 2

        nclass1 = class1.shape[0]
        nclass2 = class2.shape[0]
        print(nclass1)
        print(nclass2)
        # Class priors: in this case, we have an equal number of training
        # examples for each class, so both priors are 0.5
        prior1 = nclass1 / float(nclass1 + nclass2)
        prior2 = nclass2 / float(nclass1 + nclass1)

        mean1 = np.mean(class1, axis=0)
        mean2 = np.mean(class2, axis=0)

        class1_centered = class1 - mean1
        class2_centered = class2 - mean2

        # Calculate the covariance between the features
        cov1 = class1_centered.T.dot(class1_centered) / (nclass1 - nclasses)
        cov2 = class2_centered.T.dot(class2_centered) / (nclass2 - nclasses)
        W = (mean2 - mean1).dot(np.linalg.pinv(prior1 * cov1 + prior2 * cov2))
        b = (prior1 * mean1 + prior2 * mean2).dot(W)

        return (W, b)

    def apply_lda(self, test, W, b):
        '''
        Applies a previously trained LDA to new data.
        arguments:
            test - An array (features x trials) containing the data
            W    - The project matrix W as calculated by train_lda()
            b    - The offsets b as calculated by train_lda()
        returns:
            A list containing a classlabel for each trial
        '''
        ntrials = test.shape[1]

        prediction = []
        for i in range(ntrials):
            # The line below is a generalization for:
            # result = W[0] * test[0,i] + W[1] * test[1,i] - b
            result = W.dot(test[:, i]) - b
            if result <= 0:
                prediction.append(1)
            else:
                prediction.append(2)

        return np.array(prediction)

