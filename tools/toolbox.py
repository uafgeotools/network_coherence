from classes.network_coherence_dataclass import DataBin
import os
import warnings
import threading
from obspy.geodetics import gps2dist_azimuth
import time
import numpy as np
from scipy.signal import coherence, csd, welch
import multiprocess as mp
from obspy.core import Stream

def remove_network_response(st, inv, type='full'):
    """
        Removes the instrument response from the traces in the stream.
        Interpolates differing sample rates, if present.

        Parameters:
        st (obspy.core.stream.Stream): The stream object containing the traces to be processed.
        inv (obspy.core.inventory.inventory.Inventory): The inventory object containing the instrument response information.
        type (str): The type of response removal to perform. Options are 'full' for full response removal or any other value for sensitivity removal.

        Returns:
        obspy.core.stream.Stream: The stream object with the instrument response removed.
    """
    if type == 'full':
        print("Starting full response removal.")
        for tr in st:
            fs_resp = tr.stats.sampling_rate
            pre_filt = [0.0005, 0.001, fs_resp / 2 - 2, fs_resp / 2]
            if tr.stats.channel[1:] == 'DF' or tr.stats.channel[1:] == 'DO' or tr.stats.channel[1:] == 'DH':
                tr.remove_response(inventory=inv, pre_filt=pre_filt, output='VEL', water_level=None)
            else:
                # water level hardcoded for seismic
                tr.remove_response(inventory=inv, pre_filt=pre_filt, output='VEL', water_level=60)
    else:
        print("Starting sensitivity removal.")
        st.remove_sensitivity(inv)

    print("Response/sensitivity removal complete.")

    sampling_rates = {tr.stats.sampling_rate for tr in st}  # get all sampling rates
    if len(sampling_rates) > 1:
        # find the trace with the lowest sampling rate
        filt_tr = min(range(len(st)), key=lambda i: st[i].stats.sampling_rate)
        # pre-filter before interpolation
        st.filter('lowpass', freq=st[filt_tr].stats.sampling_rate / 2 - 2,
                  corners=12, zerophase=True)

        # interpolate to the lowest sampling rate if there are differences
        st.interpolate(sampling_rate=st[filt_tr].stats.sampling_rate, method='lanczos', a=15)

    return st

def rotate_stations(st, inv, source_lat, source_lon, output='radial'):
    """
        Rotates seismometer components to be parallel with or orthogonal to the inferred source location.

        Parameters:
        st (obspy.core.stream.Stream): The stream object containing the traces to be processed.
        inv (obspy.core.inventory.inventory.Inventory): The inventory object containing the station metadata.
        source_lat (float): The latitude of the source location.
        source_lon (float): The longitude of the source location.
        output (str): The desired output component. Component options are 'radial' or 'transverse'

        Returns:
        obspy.core.stream.Stream: The stream object with the rotated components.
    """
    rotated_st = Stream() # Initialize empty stream
    for i in range(len(inv[0])):  # Do stream rotations towards inferred source location

        # Compute distance and back azimuth from current station
        _, _, baz = gps2dist_azimuth(source_lat, source_lon, inv[0][i].latitude, inv[0][i].longitude)

        station_code = inv[0][i].code
        station_st = st.select(station=station_code, component='N') + st.select(station=station_code, component='E')

        station_st.rotate(method="NE->RT", back_azimuth=baz)
        rotated_st += station_st

    if output == 'radial':
        rotated_st = rotated_st.select(component='R')  # only keep radial component
    else:
         rotated_st = rotated_st.select(component='T')  # keep only transverse component

    return rotated_st

# --------------COHERENCE CALCULATION-----------------------------------------

def get_network_coherence(st, window_length, window_overlap, n_jobs=1):

    data = DataBin(window_length, window_overlap)
    data.build_data(st)

    # fill time vector t
    for jj in range(data.nits):
        t0_ind = data.intervals[jj]
        try:
            data.t[jj] = data.tvec[t0_ind + int(np.round(data.winlensamp / 2))]
        except Exception:
            data.t[jj] = np.nanmax(data.t)

    progress_counter = mp.Value('i', 0)  # initialize counter for progress tracking
    nPairs = int(len(st) * (len(st) - 1) / 2)  # determine number of unique station pairs

    # function to compute median network coherence for a single time window
    def compute_interval_median_Cxy2(jj):
        t0_ind = data.intervals[jj]
        tf_ind = data.intervals[jj] + data.winlensamp

        participating = set()            # stations that actually contributed to computed pairs
        Cxy2_list = []                   # list of pairwise coherence arrays

        for i in range(len(st)):
            data_i = st[i].data[t0_ind:tf_ind]
            # skip completely dead / constant traces
            if np.unique(data_i).size == 1:
                continue
            for j in range(i + 1, len(st)):
                data_j = st[j].data[t0_ind:tf_ind]
                if np.unique(data_j).size == 1:
                    continue

                # compute pair coherence
                try:
                    _, Cxy2 = coherence(data_i, data_j,
                                        fs=data.sampling_rate, window=data.window,
                                        nperseg=data.sub_window, noverlap=data.noverlap)
                except Exception:
                    # if coherence failed for this pair, skip it
                    continue

                # consider a pair "valid" only if it produced any finite values
                if np.any(np.isfinite(Cxy2)):
                    Cxy2_list.append(Cxy2)
                    participating.add(st[i].stats.station)
                    participating.add(st[j].stats.station)
                else:
                    # pair produced only NaNs (or non-finite) -> treat as non-contributing
                    continue

        # increment shared progress counter for the pool/monitor
        global progress_counter
        progress_counter.value += 1

        # compute median across pairs if any valid pairs exist
        if len(Cxy2_list) > 0:
            median_Cxy2 = np.median(np.array(Cxy2_list), axis=0)
            n_contributing_stations = len(participating)
            # If you want to treat median that is all-NaN OR all-zero as "no contributors",
            # apply that rule here (you asked for 0 when column value is 0 or nan).

            all_nan = np.all(np.isnan(median_Cxy2))  # if all values are NaN, treat as < 2 stations
            all_zero = np.all(np.isfinite(median_Cxy2) & (np.abs(median_Cxy2) <= 1e-12))  # if all values are zero (or very close), treat as < 2 stations
            if all_nan or all_zero:
                n_contributing_stations = 0
                median_Cxy2 = np.full_like(median_Cxy2, np.nan)  # represent as no data
        else:
            # no valid pairs -> no contributors; return NaN column
            median_Cxy2 = np.full_like(data.freq_vector, np.nan)
            n_contributing_stations = 0

        return median_Cxy2, n_contributing_stations

    if n_jobs > os.cpu_count():  # ensure n_jobs is not greater than the number of available cores
        n_jobs = os.cpu_count()

    # start parallel processing
    with mp.Pool(processes=n_jobs, initializer=init_worker, initargs=(progress_counter,)) as pool:
        monitor_thread = threading.Thread(target=monitor_progress, args=(data.nits, progress_counter))
        monitor_thread.start()
        results = pool.map(compute_interval_median_Cxy2, range(data.nits))
        monitor_thread.join()
        print("Processing: 100% complete")

    Cxy2_norm = np.array([res[0] for res in results]).T  # transpose to have freq x time
    n_contributing_stations = np.array([res[1] for res in results])  # get number of stations per window
    data.n_station_contributions = n_contributing_stations
    data.nPairs = nPairs

    return Cxy2_norm, data


def init_worker(counter):
    global progress_counter
    progress_counter = counter

# function to keep track of multiprocessing progress
def monitor_progress(total_tasks, progress_counter):
    milestones = [0.20, 0.40, 0.60, 0.8]
    printed = {m: False for m in milestones}
    while progress_counter.value < total_tasks:
        percentage = progress_counter.value / total_tasks
        for m in milestones:
            if not printed[m] and percentage >= m:
                print(f"Processing: {m * 100:.0f}% complete")
                printed[m] = True
                if m == milestones[-1]:  # Stop after 80%
                    return
                break
        time.sleep(1)

def get_interstation_coherograms(st, data, n_jobs=1):

    progress_counter = mp.Value('i', 0)  # initialize counter for progress tracking

    # function to compute inter-station coherence for a single station pair
    def compute_interstation_Cxy2(args):
        i, j = args
        Cxy2_list = []

        for jj in range(data.nits):
            t0_ind = data.intervals[jj]
            tf_ind = data.intervals[jj] + data.winlensamp

            _, Cxy2 = coherence(st[i].data[t0_ind:tf_ind], st[j].data[t0_ind:tf_ind],
                                fs=data.sampling_rate, window=data.window,
                                nperseg=data.sub_window, noverlap=data.noverlap)
            Cxy2_list.append(Cxy2)

        Cxy2_array = np.array(Cxy2_list).T
        station_pair = (st[i].stats.station, st[j].stats.station)

        global progress_counter
        progress_counter.value += 1  # increment progress counter

        return (station_pair, Cxy2_array)

    if n_jobs > os.cpu_count():  # ensure n_jobs is not greater than the number of available cores
        n_jobs = os.cpu_count()

    N = len(st)
    tasks = [(i, j) for i in range(N) for j in range(i + 1, N)]  # all unique station pairs
    nPairs = len(tasks)  # determine number of unique station pairs
    # start parallel processing
    with mp.Pool(processes=n_jobs, initializer=init_worker, initargs=(progress_counter,)) as pool:
        monitor_thread = threading.Thread(target=monitor_progress, args=(nPairs, progress_counter))
        monitor_thread.start()
        results = pool.map(compute_interstation_Cxy2, tasks)
        monitor_thread.join()
        print("Processing: 100% complete")
    coherograms = {pair: Cxy2 for pair, Cxy2 in results}

    return coherograms


def get_interstation_phase(st, window_length, window_overlap, n_jobs=1):
    data = DataBin(window_length, window_overlap)
    data.build_data(st)

    # fill time vector t
    for jj in range(data.nits):
        t0_ind = data.intervals[jj]
        try:
            data.t[jj] = data.tvec[t0_ind + int(np.round(data.winlensamp / 2))]
        except Exception:
            data.t[jj] = np.nanmax(data.t)

    progress_counter = mp.Value('i', 0)  # initialize counter for progress tracking

    # function to compute inter-station coherence for a single station pair
    def compute_interstation_phase(args):
        i, j = args
        phase_list = []

        for jj in range(data.nits):
            t0_ind = data.intervals[jj]
            tf_ind = data.intervals[jj] + data.winlensamp
            # get the cross-spectrum
            _, Sxy = csd(st[i].data[t0_ind:tf_ind], st[j].data[t0_ind:tf_ind],
                                fs=data.sampling_rate, window=data.window,
                                nperseg=data.sub_window, noverlap=data.noverlap)

            phase = np.angle(Sxy)  # get the phase from the cross-spectrum
            phase_list.append(phase)

        phase_array = np.array(phase_list).T
        station_pair = (st[i].stats.station, st[j].stats.station)

        global progress_counter
        progress_counter.value += 1  # increment progress counter

        return (station_pair, phase_array)

    if n_jobs > os.cpu_count():  # ensure n_jobs is not greater than the number of available cores
        n_jobs = os.cpu_count()

    N = len(st)
    tasks = [(i, j) for i in range(N) for j in range(i + 1, N)]  # all unique station pairs
    nPairs = len(tasks)  # determine number of unique station pairs

    # start parallel processing
    with mp.Pool(processes=n_jobs, initializer=init_worker, initargs=(progress_counter,)) as pool:
        monitor_thread = threading.Thread(target=monitor_progress, args=(nPairs, progress_counter))
        monitor_thread.start()
        results = pool.map(compute_interstation_phase, tasks)
        monitor_thread.join()
        print("Processing: 100% complete")
    phase_matrix = {pair: phase for pair, phase in results}

    return phase_matrix, data, nPairs

def get_all_spectrograms(st, window_length, window_overlap, n_jobs=1):
    data = DataBin(window_length, window_overlap)
    data.build_data(st)

    # fill time vector t
    for jj in range(data.nits):
        t0_ind = data.intervals[jj]
        try:
            data.t[jj] = data.tvec[t0_ind + int(np.round(data.winlensamp / 2))]
        except Exception:
            data.t[jj] = np.nanmax(data.t)

    progress_counter = mp.Value('i', 0)  # initialize counter for progress tracking

    # function to compute spectrogram for a single trace
    def compute_spectrogram(i):

        Sxx_list = []
        for jj in range(data.nits):
            t0_ind = data.intervals[jj]
            tf_ind = data.intervals[jj] + data.winlensamp

            _, Sxx = welch(st[i].data[t0_ind:tf_ind],
                           fs=data.sampling_rate, window=data.window,
                           nperseg=data.sub_window, noverlap=data.noverlap)
            Sxx_list.append(Sxx)

        Sxx_array = np.array(Sxx_list).T
        station_code = st[i].stats.station

        global progress_counter
        progress_counter.value += 1  # increment progress counter

        return (station_code, Sxx_array)

    if n_jobs > os.cpu_count():  # ensure n_jobs is not greater than the number of available cores
        n_jobs = os.cpu_count()

    N = len(st)

    # start parallel processing
    with mp.Pool(processes=n_jobs, initializer=init_worker, initargs=(progress_counter,)) as pool:
        monitor_thread = threading.Thread(target=monitor_progress, args=(N, progress_counter))
        monitor_thread.start()
        results = pool.map(compute_spectrogram, range(N))
        monitor_thread.join()
        print("Processing: 100% complete")

    spectrograms = {station: spec for station, spec in results}

    return spectrograms, data