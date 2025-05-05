import numpy as np
from scipy.fft import rfftfreq

class DataBin:
    """DataBin class for windowing information for time-frequency analysis and plotting"""
    def __init__(self, window_length, window_overlap):
        self.window_length = window_length
        self.window_overlap = window_overlap
        self.nPairs = None
        self.n_station_contributions = None

    def build_data(self, st):
        # Assumes all traces have the same sample rate and length
        self.sampling_rate = st[0].stats.sampling_rate
        self.winlensamp = int(self.window_length * self.sampling_rate)  # noqa
        # Sample increment (delta_t)
        self.sampinc = int((1 - self.window_overlap) * self.winlensamp) + 1
        # Time intervals to window data
        self.intervals = np.arange(0, len(st[0].data) - self.winlensamp, self.sampinc, dtype='int')  # noqa
        self.nits = len(self.intervals)
        # Pull time vector from stream object
        self.tvec = st[0].times('matplotlib')

        self.sub_window = int(np.round(self.winlensamp / 2))  # hardcoded to 2
        self.noverlap = int(self.sub_window * 0.5)  # hardcoded to 50%
        self.window = 'hann'
        self.freq_vector = rfftfreq(self.sub_window, 1 / self.sampling_rate)
        self.t = np.full(self.nits, np.nan)

        self.nStations = len(st)
        self.station_names = [tr.stats.station for tr in st]

    def add_plotting_info(self, freq_min, freq_max, channel_str, starttime,
                          endtime, source_name, source_lat, source_lon):
        self.freq_min = freq_min
        self.freq_max = freq_max

        self.channel_str = channel_str
        if self.channel_str == 'BHZ':
            self.channel_str = 'Vertical'
        elif self.channel_str == "BHN, BHE" or self.channel_str == "BHE, BHN":
            self.channel_str = 'Horizontal'

        self.starttime = starttime
        self.endtime = endtime
        self.n_days = (endtime - starttime) / 86400
        self.n_weeks = self.n_days / 7

        self.source_name = source_name
        self.source_lat = source_lat
        self.source_lon = source_lon


