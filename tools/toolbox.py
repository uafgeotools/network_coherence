from classes.network_coherence_dataclass import DataBin
import os
from numba import njit, prange
from rtm.travel_time import celerity_travel_time
from rtm.grid import _project_station_to_utm
import threading
from obspy.geodetics import gps2dist_azimuth
import time
import numpy as np
from scipy.signal import coherence, csd
import multiprocess as mp
from obspy.core import Stream

def remove_network_response(st, inv, type='full'):
    """
        Removes the instrument response from the traces in the stream.
        Interpolates differing sample rates, if present.
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
    
    # if more than one sample rate, interpolate to lowest fs
    if len({tr.stats.sampling_rate for tr in st}) > 1:  # if more than one sample rate, interpolate to lowest fs
        rate = min(tr.stats.sampling_rate for tr in st)
        st.filter("lowpass", freq=rate / 2 - 2, corners=12, zerophase=True) # antialiasing filter
        st.interpolate(sampling_rate=rate, method="lanczos", a=15)

    return st

def rotate_stations(st, inv, source_lat, source_lon, output='radial'):
    """
        Rotates seismometer components to be parallel with or orthogonal to the inferred source location.
    """
    rotated_st = Stream() 
    for i in range(len(inv[0])):  # Do stream rotations towards inferred source location for each station

        # Compute back azimuth from current station to inferred source
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
        
def get_network_coherence(st, window_length, window_overlap, n_jobs=1):
    """
        Computes the network-wide median coherence value across all valid and unique station pairs for each time window.
    """
    data = DataBin(window_length, window_overlap)
    data.build_data(st)

    # fill time vector t
    for jj in range(data.nits):
        t0_ind = data.intervals[jj]
        try:
            data.t[jj] = data.tvec[t0_ind + int(np.round(data.winlensamp / 2))]
        except Exception:
            data.t[jj] = np.nanmax(data.t)

    counter = mp.Value('i', 0)  # initialize counter for progress tracking
    nPairs = int(len(st) * (len(st) - 1) / 2)  # determine number of unique station pairs

    # function to compute median network coherence for a single time window
    def compute_interval_median_Cxy2(jj):
        t0_ind = data.intervals[jj]
        tf_ind = data.intervals[jj] + data.winlensamp

        participating = set()  # stations that actually contributed to computed pairs
        Cxy2_list = []  # list of pairwise coherence arrays

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
                    # pair produced only NaNs, skip it.
                    continue

        # increment shared progress counter for the pool/monitor
        with progress_counter.get_lock():
            progress_counter.value += 1

        # compute median across pairs
        if len(Cxy2_list) > 0:
            median_Cxy2 = np.median(np.array(Cxy2_list), axis=0)
            n_contributing_stations = len(participating)

            all_nan = np.all(np.isnan(median_Cxy2))
            all_zero = np.all(np.isfinite(median_Cxy2) & (np.abs(median_Cxy2) <= 1e-12))
            if all_nan or all_zero:
                n_contributing_stations = 0
                median_Cxy2 = np.full_like(median_Cxy2, np.nan)  # represent as no data
        else:
            median_Cxy2 = np.full_like(data.freq_vector, np.nan)
            n_contributing_stations = 0

        return median_Cxy2, n_contributing_stations

    if n_jobs > os.cpu_count():  # ensure n_jobs is not greater than the number of available cores
        print("Warning: n_jobs greater than available CPU cores. Setting n_jobs to max available cores -1.")
        n_jobs = os.cpu_count() -1

    # start parallel processing
    with mp.Pool(processes=n_jobs, initializer=init_worker, initargs=(counter,)) as pool:
        monitor_thread = threading.Thread(target=monitor_progress, args=(data.nits, counter))
        monitor_thread.start()
        results = pool.map(compute_interval_median_Cxy2, range(data.nits))
        monitor_thread.join()
        print("Processing: 100% complete")

    Cxy2_net = np.array([res[0] for res in results]).T  # transpose to have freq x time
    n_contributing_stations = np.array([res[1] for res in results])  # get number of stations per window
    data.n_station_contributions = n_contributing_stations
    data.nPairs = nPairs

    return Cxy2_net, data

def get_interstation_coherograms(st, data, n_jobs=1):
    """
        Computes pairwise inter-station coherograms for all unique station pairs.
    """
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

def get_interstation_phase_and_coherence(st, window_length, window_overlap, n_jobs=1):
    """
        Computes inter-station phase and coherence for all unique station pairs,
        and returns network coherence.
    """
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
    def compute_interstation_phase_and_coherence(args):
        i, j = args
        phase_list = []
        coherence_list = []

        for jj in range(data.nits):
            t0_ind = data.intervals[jj]
            tf_ind = data.intervals[jj] + data.winlensamp
            
            # get the cross-spectrum
            _, Sxy = csd(st[i].data[t0_ind:tf_ind], st[j].data[t0_ind:tf_ind],
                         fs=data.sampling_rate, window=data.window,
                         nperseg=data.sub_window, noverlap=data.noverlap)
            
            phase = np.angle(Sxy)  # get the phase from the cross-spectrum

            # get the coherence
            _, Cxy2 = coherence(st[i].data[t0_ind:tf_ind], st[j].data[t0_ind:tf_ind],
                                fs=data.sampling_rate, window=data.window,
                                nperseg=data.sub_window, noverlap=data.noverlap)

            phase_list.append(phase)
            coherence_list.append(Cxy2)

        phase_array = np.array(phase_list).T
        coherence_array = np.array(coherence_list).T
        station_pair = (st[i].stats.station, st[j].stats.station)

        global progress_counter
        progress_counter.value += 1  # increment progress counter

        return (station_pair, phase_array, coherence_array)

    if n_jobs > os.cpu_count():  # ensure n_jobs is not greater than the number of available cores
        print("Warning: n_jobs greater than available CPU cores. Setting n_jobs to max available cores -1.")
        n_jobs = os.cpu_count() -1

    N = len(st)
    tasks = [(i, j) for i in range(N) for j in range(i + 1, N)]  # all unique station pairs
    nPairs = len(tasks)

    # start parallel processing
    with mp.Pool(processes=n_jobs, initializer=init_worker, initargs=(progress_counter,)) as pool:
        monitor_thread = threading.Thread(target=monitor_progress, args=(nPairs, progress_counter))
        monitor_thread.start()
        results = pool.map(compute_interstation_phase_and_coherence, tasks)
        monitor_thread.join()
        print("Processing: 100% complete")
        
    phase_pairs = {pair: phase for pair, phase, _ in results}
    coherence_pairs = {pair: coh for pair, _, coh in results}

    coherence_stack = np.stack([c for _, _, c in results], axis=0)
    network_coherence = np.nanmedian(coherence_stack, axis=0)

    data.pairs = list(phase_pairs.keys())
    
    return phase_pairs, coherence_pairs, network_coherence, data

def compute_phi_obs(phase_pairs, coherence_pairs, data, coh_threshold=0.5):
    """
        Computes the coherence-weighted circular mean phase (phi_obs)
        and the per-frequency average coherence (coh_avg) for each station-pair.
        Return also the final weighting vector used in the grid-search misfit.
    """
    # First, we create a frequency vector corresponding to the analysis band selected
    freq_mask = (data.freq_vector >= data.freq_min) & (data.freq_vector <= data.freq_max)

    P = len(data.pairs)  # number of station pairs
    F = np.count_nonzero(freq_mask)  # number of frequencies after masking

    phi_obs = np.zeros((P, F))  # observed phase vector for each pair
    coh_avg = np.zeros((P, F))  # mean coherence for each pair

    # Loop over pairs and compute the weighted circular mean phase over time (for each frequency)
    for p, pair in enumerate(data.pairs):
        
        # Extract phase and coherence for this pair in the frequency band of interest
        phase = phase_pairs[pair][freq_mask]
        coh = coherence_pairs[pair][freq_mask]

        # Weighted complex sum across time for each frequency
        weighted_sum = np.sum(coh * np.exp(1j * phase), axis=1)
        
        # sum the coherence across time for each frequency
        weight_total = np.sum(coh, axis=1)
        
        # Circular mean phase = angle of weighted average
        phi = np.angle(weighted_sum / weight_total)
        phi_obs[p, :] = phi  # store the observed phase for this pair

        # Also compute the mean coherence per freq across time
        mean_coh = np.mean(coh, axis=1)
        coh_avg[p, :] = mean_coh


    coh_weight = coh_avg  # can be **2
    coh_weight[
        coh_avg < coh_threshold] = 0.0  # apply threshold: If a frequency has little coherence, we don't even want to trust the circular mean phase.

    return phi_obs, coh_weight

def grid_search_phase(st, grid, phase_obs, coh_weight, wave_velocity, dem, data):
    """
        Perform a grid search over candidate source locations using phase misfit
        between observed and theoretical inter-station phase differences.
    """

    S = grid

    # Project stations in processed_st to UTM if necessary
    if grid.UTM:
        for tr in st:
            tr.stats.utm_x, tr.stats.utm_y = _project_station_to_utm(tr, grid)
            tr.stats.utm_zone = grid.UTM['zone']
    
    # compute theoretical travel times via RTM function
    travel_times = celerity_travel_time(grid, st, celerity=wave_velocity, dem=dem)

    # Store celerity in S attributes
    S.attrs['celerity'] = wave_velocity

    print('----------------------')
    print('PERFORMING GRID SEARCH')
    print('(Numba-accelerated)')
    print('----------------------')

    # Frequency vector and omega for the band of interest
    freq_mask = (
        (data.freq_vector >= data.freq_min) &
        (data.freq_vector <= data.freq_max)
    )
    f_band = data.freq_vector[freq_mask]
    omega = 2.0 * np.pi * f_band

    # Precompute mapping from pair -> station indices
    idx0 = np.array([
        next(i for i, tr in enumerate(st) if tr.stats.station == p[0])
        for p in data.pairs
    ])
    idx1 = np.array([
        next(i for i, tr in enumerate(st) if tr.stats.station == p[1])
        for p in data.pairs
    ])

    tic = time.time()

    travel_times_data = travel_times.data

    # Compute misfit grid 
    misfit_grid = compute_phase_misfit_grid(travel_times_data, idx0, idx1,
                                            omega, phase_obs, coh_weight)

    # Assign result back to xarray grid
    S.data = misfit_grid

    toc = time.time()
    print(f'Done (elapsed time = {toc - tic:.1f} s)')

    return S

@njit(parallel=True, fastmath=True, nogil=True)
def compute_phase_misfit_grid(travel_times, idx0, idx1, omega, phase_obs, coh_weight):
    """
        Computes the phase-misfit value at each grid point using precomputed travel
        times, station-pair indices, observed phase, and coherence weights.
    """
    n_stations, ny, nx = travel_times.shape
    n_pairs, n_freqs = phase_obs.shape
    misfit_grid = np.empty((ny, nx), dtype=np.float64)

    for i in prange(nx):  # parallelize outer loop
        for j in range(ny):
            
            # travel-time difference for all station pairs
            t_all = travel_times[:, j, i]
            delta_t = t_all[idx0] - t_all[idx1]

            mean_misfit = 0.0
            weight_sum = 0.0
            for p in range(n_pairs):
                for f in range(n_freqs):
                    # theoretical phase for this pair and frequency
                    phi_theo = (delta_t[p] * omega[f]) % (2 * np.pi)
                    # circular difference between observed and theoretical phase
                    res = np.angle(np.exp(1j * (phase_obs[p, f] - phi_theo)))
                    mean_misfit += coh_weight[p, f] * abs(res)
                    weight_sum += coh_weight[p, f]

            misfit_grid[j, i] = mean_misfit / weight_sum if weight_sum > 0 else np.nan

    return misfit_grid