import numpy as np

class DataBin:
    """DataBin class for windowing information for time-frequency analysis and plotting"""
    def __init__(self, window_length, window_overlap):
        self.window_length = window_length
        self.window_overlap = window_overlap

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

        self.station_names = [tr.stats.station for tr in st]


