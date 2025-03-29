from classes.network_coherence_dataclass import DataBin
import os
import threading
from obspy.geodetics import gps2dist_azimuth
import time
import numpy as np
from scipy.signal import coherence
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
                tr.remove_response(inventory=inv, pre_filt=pre_filt, output='DISP', water_level=60)
    else:
        print("Starting sensitivity removal.")
        st.remove_sensitivity(inv)

    print("Response/sensitivity removal complete.")

    # Here we do secondary processing and interpolate stream if stations have differing sampling rates
    st.detrend('linear')

    # find the trace with the lowest sampling rate
    filt_tr = min(range(len(st)), key=lambda i: st[i].stats.sampling_rate)
    # pre-filter before interpolation
    st.filter('lowpass', freq=st[filt_tr].stats.sampling_rate / 2 - 2, corners=12, zerophase=True)

    sampling_rates = {tr.stats.sampling_rate for tr in st}  # get all sampling rates
    if len(sampling_rates) > 1:
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

        Cxy2_list = []
        for i in range(len(st)):
            for j in range(i + 1, len(st)):
                _, Cxy2 = coherence(st[i].data[t0_ind:tf_ind], st[j].data[t0_ind:tf_ind],
                                   fs=data.sampling_rate, window=data.window,
                                   nperseg=data.sub_window, noverlap=data.noverlap)
                Cxy2_list.append(Cxy2)
        global progress_counter
        progress_counter.value += 1  # increment progress counter

        # This returns the median coherence for a single time window for all unique station pairs.
        return np.median(Cxy2_list, axis=0)

    if n_jobs > os.cpu_count():  # ensure n_jobs is not greater than the number of available cores
        n_jobs = os.cpu_count()

    # start parallel processing
    with mp.Pool(processes=n_jobs, initializer=init_worker, initargs=(progress_counter,)) as pool:
        monitor_thread = threading.Thread(target=monitor_progress, args=(data.nits, progress_counter))
        monitor_thread.start()
        results = pool.map(compute_interval_median_Cxy2, range(data.nits))
        monitor_thread.join()

    Cxy2_norm = np.array(results).T  # transpose to have time on the x-axis

    return Cxy2_norm, data, nPairs

def init_worker(counter):
    global progress_counter
    progress_counter = counter

# function to keep track of multiprocessing progress
def monitor_progress(total_tasks, progress_counter):
    milestones = [0.25, 0.50, 0.75]
    printed = {m: False for m in milestones}
    while progress_counter.value < total_tasks:
        percentage = progress_counter.value / total_tasks
        for m in milestones:
            if not printed[m] and percentage >= m:
                print(f"Coherence processing: {m * 100:.0f}% complete")
                printed[m] = True
                if m == milestones[-1]:  # Stop after 75%
                    return
                break
        time.sleep(1)