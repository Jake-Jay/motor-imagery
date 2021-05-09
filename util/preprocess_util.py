from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import mne
import pdb



# -----------------------------------------------------------------------
# Preprocessing 
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


def rereference(samples_buffer, channel=None):
    '''Rerefence all data wrt channel
    '''
    if channel == None:
        return samples_buffer - np.mean(samples_buffer, axis=0)

    return samples_buffer - samples_buffer[channel]


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
        - ftype: filter type
    Returns:
        - filtered data of shape (num_channels)(buffer_size)
    '''
    # Check type
    try:
        assert ftype in ['fir', 'iir'], 'You did not specify the type correctly'
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

def visualise_filter(filter_params, fs:int):
    w1, h1 = signal.freqz(b_bp, fs=fs)
    plt.plot(w1, 20*np.log10(np.abs(h1)), 'b')

    plt.show()

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
