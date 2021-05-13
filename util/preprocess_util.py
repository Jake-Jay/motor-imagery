from scipy import signal
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import mne
import pdb


# -----------------------------------------------------------------------
# Wrapper Functions 
# -----------------------------------------------------------------------


def preprocess(samples_buffer, use_mne=False, **kwargs):
    '''All the preprocessing and filtering

    - Rereference
    - Filter (bandpass filter and notch)
    '''

    if use_mne:
        mne_buffer = np2mne(samples_buffer, fs=kwargs['fs'])
        preprocessed_buffer = filt_mne(
            mne_buffer,
            lf=kwargs['lf'],
            hf=kwargs['hf'],
            nf=kwargs['nf']
        )

        # preprocessed_buffer.plot_psd(
        #     show=True,
        # )

        preprocessed_buffer = mne2raw(preprocessed_buffer)
    else:
        '''Using scipy filtering'''

        # Rereference wrt 1st Unicorn electrode
        samples_buffer = rereference(
            samples_buffer,
            channel=None
        )

        preprocessed_buffer = filt(
            samples_buffer,
            lf=kwargs['lf'],
            hf=kwargs['hf'],
            nf=kwargs['nf'],
            fs=kwargs['fs']
        )   

    return preprocessed_buffer


# -----------------------------------------------------------------------
# Standardize and reduce noise
# -----------------------------------------------------------------------

def rereference(samples_buffer, channel=None):
    '''Rerefence all data wrt channel
    '''
    if channel == None:
        return samples_buffer - np.mean(samples_buffer, axis=0)

    return samples_buffer - samples_buffer[channel]

def normalise_eeg(eeg_data):
    '''Normalise each channel to have std of 1'''
    return eeg_data / eeg_data.std(axis=1)[:, None]

# -----------------------------------------------------------------------
# Filterings 
# -----------------------------------------------------------------------

def filt(samples_buffer, lf=8, hf=30, nf=50, fs=250):
    '''Apply a bandpass and notch filter
    
    Params:
        - samples_buffer: data of shape (num_channels)(buffer_size)
        - lf: low frequency cutoff
        - hf: high frequency cutoff
        - nf: notch frequency (set to None if you do not want to notch filt)
        - fs: sampling frequency
    
    Returns:
        - filtered data of the same shape as samples_buffer

    '''

    filtered_buffer = bandpass(
        samples_buffer,
        lf=lf,
        hf=hf,
        ftype='iir',
        fs=fs
    )

    # Apply the filter
    if nf is not None:
        filtered_buffer = notch(
            filtered_buffer,
            fs=fs,
            notch_f=nf
        )

    return filtered_buffer

def bandpass(samples_buffer: np.ndarray, lf=4, hf=30, fs=250, ftype:str='fir'):
    '''Apply a bandpass filter
    
    Params:
        - samples_buffer: raw array of shape (num_channels)(buffer_size)
        - lf: low frequency cuttoff
        - hf: high frequency cuttoff
        - fs: sampling frequency
        - ftype: filter type [fir, iir, butter]
    Returns:
        - filtered data of shape (num_channels, buffer_size)
    '''
    # Check type
    try:
        assert ftype in ['fir', 'iir', 'butter'], 'You did not specify the type correctly'
    except Exception as e:
        raise

    if ftype == 'iir':
        # a, b = signal.iirfilter(6, [lf/(fs/2.0), hf/(fs/2.0)])
        a, b = signal.iirfilter(
            6, 
            [lf, hf],
            fs=fs
        )
        filtered_data = signal.filtfilt(a, b, samples_buffer, axis=1)

    elif ftype == 'fir':
        b_bp = signal.firwin(
            21, 
            [lf, hf],
            width=0.05, 
            fs=fs, 
            pass_zero='bandpass'
        )

        filtered_data = signal.lfilter(b_bp, [1], samples_buffer, axis=1)
    
    elif ftype == 'butter':
        b, a = signal.butter(4, Wn=[lf, hf], btype='bandpass', fs=fs)

        filtered_data = signal.filtfilt(b, a, samples_buffer, padlen=150)
    
    return filtered_data

def notch(samples_buffer:np.ndarray, notch_f:int=50, Q:int=30, fs:int=250):
    '''Apply notch filter

    Params:
        - samples_buffer: raw array of shape (num_channels)(buffer_size)
        - notch_f: notch filter frequency
        - Q: quality factor
    Returns:
        - filtered data of shape (num_channels)(buffer_size)
    '''

    b, a = signal.iirnotch(w0=notch_f, Q=30, fs=fs)
    filtered_data = signal.filtfilt(b, a, samples_buffer, axis=1)

    # pdb.set_trace()

    # freq, h = signal.freqz(b, a, fs=fs)

    # # Plot
    # plt.figure()
    # plt.plot(freq, 20*np.log10(abs(h)), color='blue')
    # plt.show()
    return samples_buffer


# -----------------------------------------------------------------------
# Channels by name or by channel number
# -----------------------------------------------------------------------
def channel2name(channel_number):
    mapping = {
        0:  'Fz',    
        1:  'C3', 
        2:  'Cz', 
        3:  'C4',
        4:  'Pz', 
        5:  'PO7', 
        6:  'Oz', 
        7:  'PO8',
    }
    return mapping[channel_number]

def name2channel(name):
    mapping = {
        'Fz': 0,    
        'C3': 1, 
        'Cz': 2, 
        'C4': 3,
        'Pz': 4, 
        'PO7': 5,
        'Oz': 6, 
        'PO8': 7,
    }
    return mapping[name]

def get_channel(channel_name:str, eeg_data):
    '''
    Params:
        - channel_name: One of [Fz, C3, Cz, C4, Pz, P07, Oz, PO8]
        - eeg_data: shape (num_channels, num_samples)
    Returns:
        - single channel
    '''
    return eeg_data[name2channel[channel_name], :]


# -----------------------------------------------------------------------
# CSP (common spatial pattern) Related Methods
# -----------------------------------------------------------------------
class CSP:

    def __init__(self, trials, classes, nsamples, nchannels, W=None) -> None:
        self.nsamples = nsamples
        self.nchannels = nchannels
        self.trials = trials

        if W is None:
            self.W = self.csp(trials[classes[0]], trials[classes[1]])
        else:
            self.W = W

        self.trials_csp = {
            classes[0]: self.apply_mix(self.W, self.trials[classes[0]]),
            classes[1]: self.apply_mix(self.W, self.trials[classes[1]])
        }

    def get_csp_trials(self):
        return self.trials_csp
    
    def get_W(self):
        return self.W

    def cov(self, trials):
        ''' Calculate the covariance for each trial and return their average '''
        ntrials = trials.shape[2]
        covs = [ trials[:,:,i].dot(trials[:,:,i].T) / self.nsamples for i in range(ntrials) ]
        return np.mean(covs, axis=0)

    def whitening(self, sigma):
        ''' Calculate a whitening matrix for covariance matrix sigma. '''
        U, l, _ = linalg.svd(sigma)
        return U.dot( np.diag(l ** -0.5) )

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
        B, _, _ = linalg.svd( P.T.dot(cov_f).dot(P) )
        W = P.dot(B)
        return W

    def apply_mix(self, W, trials):
        ''' Apply a mixing matrix to each trial (basically multiply W with the EEG signal matrix)'''
        ntrials = trials.shape[2]
        trials_csp = np.zeros((self.nchannels, self.nsamples, ntrials))
        for i in range(ntrials):
            trials_csp[:,:,i] = W.T.dot(trials[:,:,i])
        return trials_csp

# -----------------------------------------------------------------------
# MNE Preprocessing
# -----------------------------------------------------------------------

def np2mne(samples_buffer, fs=250):
    '''Convert numpy array to mne RawArray

    Hardcode the channel names & types based on prior knowledge.
    Remove the accelerometer channels. The first 8 channels of the
    Unicorn stream contain the EEG data

    Params:
        - samples_buffer: numpy array of shape (num_channels)(buffer_len)
    '''

    ch_names = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8']
    ch_types = ['eeg' for _ in range(len(ch_names))]
    info = mne.create_info(
        ch_names=ch_names, 
        ch_types=ch_types,
        sfreq=fs
    )

    return mne.io.RawArray(samples_buffer[0:8, :], info)

def mne2raw(mne_buffer):
    '''Convert numpy array to mne RawArray'''
    return mne_buffer._data

def filt_mne(mne_array, lf, hf, nf):
    '''Use MNE library to apply a bandpass and notch filter
    
    Params:
        - mne_array: mne array containing data of shape (num_channels)(buffer_size)
        - lf: low frequency cutoff
        - hf: high frequency cutoff
        - nf: notch frequency (set to None if you do not want to notch filt)
        - fs: sampling frequency
    
    Returns:
        - filtered data of the same shape as samples_buffer
    '''

    filtered_data = mne_array.filter(
        l_freq=lf,
        h_freq=hf,
        method='iir'
    )

    filtered_data = filtered_data.notch_filter(
        freqs=nf,
        method='iir'
    )

    return filtered_data
