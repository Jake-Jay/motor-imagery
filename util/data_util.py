import pyxdf
import numpy as np
import pdb


class EEG_Dataset():

    def __init__(self, datafile) -> None:
        
        # Get streams
        streams, header = self.load_from_xdf(datafile)

        # Extract EEG data
        try:
            eeg_stream = self.find_stream('Unicorn', streams)
        except Exception as e:
            exit()
        self.fs = int(eeg_stream['info']['nominal_srate'][0]) # 250 Hz
        self.eeg_data = eeg_stream['time_series'][:,0:8].T
        self.eeg_timestamps = eeg_stream['time_stamps']

        # Extract the marker stream (as labels)
        try:
            marker_stream = self.find_stream('openvibeMarkers', streams)
        except Exception as e:
            exit()
        marker_stream['time_series'] = np.array(marker_stream['time_series'])
        self.event_labels = [x[0] for x in marker_stream['time_series']]

        # Convert marker stream from labels to integer codes
        self.label2code = self.build_unique_mapping(self.event_labels)
        self.event_codes = np.array([self.label2code[x] for x in self.event_labels])

        # Get event time stamps
        self.event_timestamps = np.array(marker_stream['time_stamps'])
        assert self.event_timestamps.shape == self.event_codes.shape

        # Some feedback to complete the loading process
        print(f'Data succesfullly loaded from {datafile}')

    def load_from_xdf(self, datafile):
        streams, header = pyxdf.load_xdf(datafile)
        return streams, header

    def find_stream(self, name, streams):
        for stream in streams:
            if stream['info']['name'][0] == name:
                return stream
        print(f'Error: stream "{name}" not found')
        raise Exception

    def build_unique_mapping(self, event_labels):

        def _unique(input_list):
            unique_items = []
            for x in input_list:
                if x not in unique_items:
                    unique_items.append(x)
            return unique_items

        unique_labels = _unique(event_labels)

        unique_mapping = {}
        for i, label in enumerate(unique_labels):
            unique_mapping[label] = i
        return unique_mapping

    def build_event_timeseries(self):

        def _find_closest_time(timestamp, eeg_timestamps):
            return np.where(eeg_timestamps > timestamp)[0][0]

        eeg_start_time = self.eeg_timestamps[0]
        label_start_time = self.event_timestamps[0]
        experiment_start_time = min(label_start_time, eeg_start_time)

        relative_eeg_timestamps = self.eeg_timestamps - experiment_start_time
        relative_event_timestamps = self.event_timestamps - experiment_start_time

        # Get indices of events
        indices = [0]

        for timestamp in relative_event_timestamps:
            indices.append(_find_closest_time(timestamp, relative_eeg_timestamps))

        # Build event time series (use -1 since 0 is actually a code for no blinking)
        event_timeseries = np.zeros_like(self.eeg_timestamps) - 1

        for i, (_, code) in enumerate(zip(indices[1:], self.event_codes)):
            idx = i+1
            if idx < len(indices)-1:
                event_timeseries[indices[idx]:indices[idx+1]] = code
            else:
                event_timeseries[indices[idx]:] = code

        return relative_eeg_timestamps, relative_event_timestamps, event_timeseries



def get_trials(eeg_data, event_time_series, class_labels, nsamples:list, label2code:dict):
    '''
    Split EEG data into trials based on class labels.

    Returns:
        - dict of trials of the form {label: (nchannels)(window_len)(ntrials)}
    '''
    assert len(class_labels) == len(nsamples)

    # Dictionary to store the trials in, each class gets an entry
    trials = {}
    nchannels = eeg_data.shape[0]

    for cl_lab, nsamples in zip(class_labels, nsamples):
        event_onset_mask =  np.roll(event_time_series,1)!=event_time_series
        class_onset_mask = event_time_series == label2code[cl_lab]
        class_onset_indices = np.where(
            np.logical_and(event_onset_mask, class_onset_mask))[0]

        # Allocate memory for the trials
        trials[cl_lab] = np.zeros((nchannels, nsamples, len(class_onset_indices)))

        # Extract each trial
        for i, onset in enumerate(class_onset_indices):
            trials[cl_lab][:,:,i] = eeg_data[:, onset:onset+nsamples]
    
    return trials


def get_good_trials(trials, nchannels, class_nsamples, upperlimit_std, upperlimit_max):

    classes = list(trials.keys())
    good_trials = {}

    good_trials = {}

    for cl, nsamples in zip(classes, class_nsamples):
        good_trials[cl] = np.zeros((nchannels, nsamples,1))

    ntrials = trials[classes[0]].shape[2]

    for trial in range(ntrials):

        for cl in classes:
            eeg_sample = trials[cl][:,:,trial]
            if (eeg_sample.std(axis = (0,1)) < upperlimit_std[cl] and 
                eeg_sample.max(axis = (0,1)) < upperlimit_max[cl]):
                good_trials[cl] = np.concatenate(
                    [good_trials[cl], np.expand_dims(eeg_sample, -1)], 
                    axis=2
                )

    for cl in classes:
        good_trials[cl] = good_trials[cl][:,:,1:]
    
    return good_trials




