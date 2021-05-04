# Functions to get data into the traning and inferencing pipelines
import numpy as np
from mne.io import read_raw_gdf, read_raw_edf, RawArray
from pylsl import StreamInlet, resolve_stream, local_clock
from pyxdf import load_xdf, match_streaminfos, resolve_streams
from mne import Epochs, pick_types, events_from_annotations, create_info


class EEGDataAdapter:
    """
      Class for laoding data into the TUMBrain pipeline both in offline and online mode. Currently capable of reading xdf, gdf,
      edf using the graz paradigm and the unicorn headset.

      Attributes
      ----------
      device_type: str
        Specify the type of the device. For example this could be Unicorn or Smarting.

      channel_indices: list<int>
        Specity which channels of the headset have to selected to be passed onto the next stages of the pipeline.

      number_of_samples: int
        Number of samples read into each buffer (window_size)

      mode: str
        Mode of operation. Online or Offline

      inlet: StreamInlet
        From the pylsl library. The inlet recieves objects of type StreamInlet defined in the pylsl class.

      samples_buffer: list<list<float>>
        FIFO (First In First Out) buffer of last window of data.

      timestamps_buffer: list<list<integers>>
        Corresponding timestamps of FIFO (First In First Out) buffer of last window of data.

      file_type: str
        Type of file being used for the offline mode of the pipeline. For example xdf, gdf, edf

      recording_paradigm: str
        The paradigm used while Recording data for use in the offline case. For example Graz paradigm is a famous one.
        One could design their own paradigm and incorporate into the codebase accordinlgy

      event_dict: dict<str:int>
          Mapping of the event names to an integer value

      events: ndarray, shape (n_events, 3)
        The events. 1st Column timstamp, 2nd Column weird parameter that's always zero. 3rd Column is the label.

    """

    def __init__(self, device_type='Unicorn', channel_indices=[1, 2, 3, 4, 5, 6, 7, 8], mode='offline',
                 number_of_samples=125, event_dict=None):
        """ Constructor for the EEGDataAdapter class.

        Parameters
        ----------
        device_type : str, optional
            The device used for Recording or streaming data, by default 'Unicorn'. Suuports:['Unicorn, 'Graz]
        channel_indices : list, optional
            Channels in the device required for the next stages of the pipeline. Specify the channel required
            by mentioning their number in the list, by default [1,2,3,4,5,6,7,8]
        mode : str, optional
            The operation mode of the adapter. 'online' and 'offline' are the two modes of operation, by default 'offline'
        number_of_samples : int, optional
            The window size of data read from the inlet stream, by default 125
        """
        self.device_type = device_type
        self.channel_indices = channel_indices
        self.number_of_samples = number_of_samples
        self.mode = mode
        self.event_dict = event_dict

        if self.mode == 'online':
            streams = resolve_stream('type', 'EEG')
            # create a new inlet to read from the stream
            self.inlet = StreamInlet(streams[0])
            self.samples_buffer = []
            self.timestamps_buffer = []

    def online_streamer(self):
        """This function is responsible for streaming the data in an online fashion. Currently supports device type Unicorn.
        Can be extended by adding a new case for different device type. Can only be used in online mode. Function returns -1
        if the input streamer doesn't give chunks as input.
        If buffer is already filled with at least some chunk, in case of connection loss, the same chunk remains as input as it is not
        discarded.

        Returns
        -------
        samples_buffer_numpy: ndarray (n_channels, n_samples)
          returns an ndarray if there is no IndexError (chunks not of proper size). Otherwise None

        timestamps_buffer_numpy: ndarray (n_samples)
          returns an ndarray if there is no IndexError (chunks not of proper size). Otherwise None

        errorCode: integer
            0: everything OK
           -1: Stream Interuppted

        """
        if self.device_type == 'Unicorn':
            print("[INFO]: Reading from a Unicorn Headset...")
            sample, timestamp = self.inlet.pull_chunk(max_samples=self.number_of_samples)
            recv_stamp = local_clock()
            self.samples_buffer += sample
            self.timestamps_buffer += timestamp
            self.samples_buffer = self.samples_buffer[-self.number_of_samples:]
            self.timestamps_buffer = self.timestamps_buffer[-self.number_of_samples:]
            self.timestamps_buffer = [x - recv_stamp for x in self.timestamps_buffer]
            try:
                samples_buffer_numpy, timestamps_buffer_numpy = np.array(self.samples_buffer)[:,
                                                                self.channel_indices].T, np.array(
                    self.timestamps_buffer)
            except IndexError:
                print("[INFO]: Waiting for chunks.... ")
                errorCode = -1  # error code -1 -> Stream not recieved
                return None, None, errorCode
            print("[INFO]: Chunk Recieved")
            errorCode = 0  # error code 0 -> Stream OK
            return samples_buffer_numpy, timestamps_buffer_numpy, errorCode

    def read_raw_xdf(self, filename):
        """Read XDF file. Function modified from MNElab. https://github.com/cbrnr/mnelab/blob/master/mnelab/io/xdf.py.
          Reads stream of type 'EEG' and type 'Markers'.

        Parameters
        ----------
        filename : str
            Name of the XDF file (without the extension).

        Returns
        -------
        raw : mne.io.Raw
            XDF file data.
        """
        streams, _ = load_xdf(filename)
        for stream in streams:
            if stream["info"]["type"][0] == 'Data':  # Graz_data here search for 'EEG'
                break  # stream found

        n_chans = int(stream["info"]["channel_count"][0])
        fs = float(stream["info"]["nominal_srate"][0])
        labels, types, units = [], [], []
        try:
            for ch in stream["info"]["desc"][0]["channels"][0]["channel"]:
                labels.append(str(ch["label"][0]))
                if ch["type"]:
                    types.append(ch["type"][0])
                if ch["unit"]:
                    units.append(ch["unit"][0])
        except (TypeError, IndexError):  # no channel labels found
            pass
        if not labels:
            labels = [str(n) for n in range(n_chans)]
        if not units:
            units = ["NA" for _ in range(n_chans)]
        info = create_info(ch_names=labels, sfreq=fs, ch_types="eeg")
        # convert from microvolts to volts if necessary
        scale = np.array([1e-6 if u == "µV" else 1 for u in units])
        raw = RawArray((stream["time_series"] * scale).T, info)
        raw._filenames = [filename]
        first_samp = stream["time_stamps"][0]
        for stream in streams:
            if stream["info"]["type"][0] == 'Markers':
                break
        onsets = stream["time_stamps"] - first_samp
        descriptions = [item for sub in stream["time_series"] for item in sub]
        raw.annotations.append(onsets, [0] * len(onsets), descriptions)
        return raw

    def load_recorded_eeg_data_file(self, file_name="../data/B0101T", file_type='gdf', recording_paradigm='', tmin=0,
                                    tmax=2):
        """Load Data and events from a File to Numpy Arrays. The Loading depends on the FileType, Device and Recording Paradigm.


        Parameters
        ----------
        file_name : str, optional
            Filename including the relative Path, by default "../data/B0101T"
        file_type : str, optional
            file type (suffix), by default 'gdf'. Supports: [edf, gdf, xdf]
        recording_paradigm : str, optional
            Definition of the Recording paradigm, to distingiush different mappings and Channel Names, by default 'Graz'. Supports: ['Graz']
        tmin : float, optional
            Start time before event. , by default 0
        tmax : float, optional
            End time after event, by default 2

        Returns
        -------
        data : ndarray (n_channels, n_samples)
            The Data od Channels and EEG Samples as Numpy Array
        lables : ndarray (n_samples,)
            Vektor of Labels to the corresponding Data
        """
        self.file_type = file_type
        self.recording_paradigm = recording_paradigm
        self.event_dict == None

        if recording_paradigm == 'Graz':
            print("[INFO]: The Paradigm followed is the Graz paradigm")

            if self.event_dict == None:
                print("went into none")
                self.event_dict = {'noblink': 0, 'break': 1, 'imagery_handR': 2, 'imagery_handL': 3, 'imagery_foot': 4,
                                   'imagery_rotation': 5}
        print(self.event_dict)
        if recording_paradigm == 'SSVEP':
            print("[INFO]: The Paradigm followed is the Graz paradigm")
            if self.event_dict == None:
                print("went into none")
                self.event_dict = {'fs_5': 0, 'fs_10': 1, "fs_15": 1}

        if file_type == 'edf':
            print("[INFO]: Opening a edf file")
            raw_data = read_raw_edf(file_name + "." + file_type, preload=True)
            if self.device_type == 'Unicorn':
                print("[INFO]: Device type is Unicorn")
                raw_data.rename_channels(
                    {'Channel 1': 'EEG Fz', 'Channel 2': 'EEG C3', 'Channel 3': 'EEG Cz', 'Channel 4': 'EEG C4',
                     'Channel 5': 'EEG Pz', 'Channel 6': 'EEG PO7', 'Channel 7': 'EEG Oz', 'Channel 8': 'EEG PO8',
                     'Channel 9': 'Accelerometer X', 'Channel 10': 'Accelerometer Y', 'Channel 11': 'Accelerometer Z',
                     'Channel 12': 'Gyroscope X',
                     'Channel 13': 'Gyroscope Y', 'Channel 14': 'Gyroscope Z', 'Channel 15': 'Battery LeveL',
                     'Channel 16': 'Counter',
                     'Channel 17': 'Validation Indi', })

                event_strings = ['hand/left', 'hand/right', 'foot', 'noblink']
                self.select_events(selected_events=event_strings, raw_type=file_type)
            else:
                print("[ERROR]: Invalid Device type, define Events")
                events_from_annotations(raw_data)


        elif file_type == 'gdf':
            print("[INFO]: Opening a gdf file")
            raw_data = read_raw_gdf(file_name + "." + file_type, preload=True)
            events_from_annotations(raw_data)
            if self.device_type == 'Unicorn':
                print("[INFO]: Device type is Unicorn")
                raw_data.rename_channels(
                    {'Channel 1': 'EEG Fz', 'Channel 2': 'EEG C3', 'Channel 3': 'EEG Cz', 'Channel 4': 'EEG C4',
                     'Channel 5': 'EEG Pz', 'Channel 6': 'EEG PO7', 'Channel 7': 'EEG Oz', 'Channel 8': 'EEG PO8',
                     'Channel 9': 'Accelerometer X', 'Channel 10': 'Accelerometer Y', 'Channel 11': 'Accelerometer Z',
                     'Channel 12': 'Gyroscope X',
                     'Channel 13': 'Gyroscope Y', 'Channel 14': 'Gyroscope Z', 'Channel 15': 'Battery LeveL',
                     'Channel 16': 'Counter',
                     'Channel 17': 'Validation Indi', })
                event_strings = ['imagery_handL', 'imagery_handR', 'noblink', 'imagery_foot']
                self.select_events(selected_events=event_strings, raw_type=file_type)

            elif self.device_type == 'Graz':
                print("[INFO]: Device type is Unknown, but the one used from Graz")
                event_strings = ['imagery_handL', 'imagery_handR', 'noblink', 'imagery_rotation']
                self.select_events(selected_events=event_strings, raw_type=file_type)

            else:
                print("[ERROR]: Invalid Device type, define Events")
                events_from_annotations(raw_data)

        elif file_type == 'xdf':
            print("[INFO]: Opening an xdf file")
            raw_data = self.read_raw_xdf(file_name + "." + file_type)
            event_strings = list(self.event_dict.keys())
            event_strings = ['break', 'imagery_handL', 'imagery_handR']
            self.select_events(selected_events=event_strings, raw_type='xdf')


        else:
            print("[ERROR]: Invalid file type")

        self.events, _ = events_from_annotations(raw_data, event_id=self.event_dict)
        picks = pick_types(raw_data.info, misc=False, eeg=True)[self.channel_indices]
        epochs = Epochs(raw_data, self.events, self.event_dict, tmin=tmin, tmax=tmax, proj=False, baseline=None,
                        preload=True, picks=picks)
        data_mne_to_numpy_array = epochs.get_data()

        # Black magic (Reshaping numpy array obtained from MNE for our purposes)
        data = data_mne_to_numpy_array.transpose(1, 0, 2).flatten().reshape(data_mne_to_numpy_array.shape[1], -1)
        batch_with_same_label_length = data_mne_to_numpy_array.shape[2]
        labels = np.repeat(epochs.events[:, -1], batch_with_same_label_length)
        print("[INFO]: Acquired  Data and labels")
        return data, labels

    def select_events(self, selected_events=['break', 'imagery_handR', 'imagery_foot', 'imagery_handL'],
                      raw_type='xdf'):
        """This function selects the specified events and creates an event dictonary. It also unifies the mapping of events to strings
        for the different types(xdf and edf takes in strings, but e.g. gdf takes only integers). It needs a list of strings with the
        names of the Event and creates a dictonary.

        Parameters
        ----------
        selected_events : list<str>, optional
            List of event names to be selected, by default ['break', 'imagery_handR', 'imagery_foot', 'imagery_handL']

        raw_type : str, optional
            The Type of the Raw data, on which the mapping depends, by default 'xdf'

        """

        selected_event_dict = dict()

        if raw_type == 'gdf':
            # Explain dictionary numbering later
            if 'noblink' in str(selected_events):
                selected_event_dict['277'] = self.event_dict['noblink']

            if 'idle' in str(selected_events):
                selected_event_dict['276'] = self.event_dict['break']

            if 'imagery_handR' in str(selected_events):
                selected_event_dict['770'] = self.event_dict['imagery_handR']

            if 'imagery_handL' in str(selected_events):
                selected_event_dict['769'] = self.event_dict['imagery_handL']

            if 'imagery_foot' in str(selected_events):
                selected_event_dict['1095'] = self.event_dict['imagery_foot']

            if 'imagery_rotation' in str(selected_events):
                selected_event_dict['1079'] = self.event_dict['imagery_rotation']

        elif raw_type == 'xdf':
            print(self.event_dict)
            for ev in selected_events:
                selected_event_dict[ev] = self.event_dict[ev]

        elif raw_type == 'edf':
            for ev in str(selected_events):
                selected_event_dict[ev] = self.event_dict[ev]

        print("[INFO]: selected events:" + str(selected_events))
        self.event_dict = selected_event_dict

    def select_data_of_label_class(self, data, labels, selected_label):
        """ Selects data of a particular label. This function could be used to classify only specific label type.

        Parameters
        ----------
        data: ndarray (n_channels, n_samples)
            Input data matrix that needs to be passed to be processed.

        labels: ndarray (n_samples, )
            Vector of Lables to the corresponding data

        selected_label: int
            The label class that needs to be selected out from the data numpy array.

        Returns
        -------
        data_with_selected_label: ndarray (n_channels, n_samples(of particular type))
            data with labels of particular label_class.
        """
        indices = np.argwhere(labels == selected_label)
        data_with_selected_label = data[:, indices].squeeze()
        return data_with_selected_label


if __name__ == "__main__":
    loader = EEGDataAdapter('Graz', channel_indices=[1, 2, 3, 4], mode='offline')
    if loader.mode == 'online':
        while (True):
            samples, timestamps = loader.online_streamer()

    if loader.mode == 'offline':
        data, labels = loader.load_recorded_eeg_data_file(file_name="../data/B0101T", file_type='gdf')

