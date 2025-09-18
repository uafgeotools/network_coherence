from tqdm import tqdm
from obspy.geodetics import gps2dist_azimuth
import numpy as np


def meters_to_degrees(grid_spacing_m, lat_deg):

    lat = np.asarray(lat_deg, dtype=float)
    lat_rad = np.deg2rad(lat)

    # meters per degree latitude/longitude (WGS84 ellipsoid approx)
    m_per_deg_lat = (111132.954
                     - 559.822 * np.cos(2 * lat_rad)
                     + 1.175 * np.cos(4 * lat_rad)
                     - 0.0023 * np.cos(6 * lat_rad))
    m_per_deg_lon = (111412.84 * np.cos(lat_rad)
                     - 93.5 * np.cos(3 * lat_rad)
                     + 0.118 * np.cos(5 * lat_rad))

    # convert spacing in meters to degrees
    dlat = grid_spacing_m / m_per_deg_lat
    eps = 1e-9
    dlon = np.where(np.abs(m_per_deg_lon) < eps, np.nan, grid_spacing_m / m_per_deg_lon)

    return dlat, dlon

def compute_phi_obs_and_weights(phase_matrix, coherograms, data, coh_threshold=0.5):
    """
    Compute the coherence-weighted circular mean phase (phi_obs)
    and the per-frequency average coherence (coh_avg) for each station-pair.
    Return also the final weighting vector coh_weight = coh_avg ** 2 used in the grid-search misfit.
    """
    # First, we create a frequency vector corresponding to the analysis band selected
    freq_mask = (data.freq_vector >= data.freq_min) & (data.freq_vector <= data.freq_max)

    # inter-station pairs for indexing into phase_matrix and coherograms
    pairs = list(phase_matrix.keys())
    P = len(pairs)                         # number of station pairs
    F = np.count_nonzero(freq_mask)        # number of frequencies after masking

    # Preallocate outputs: rows = pairs, cols = frequency
    phi_obs = np.zeros((P, F))  # observed phase vector for each pair
    coh_avg = np.zeros((P, F))  # mean coherence for each pair

    # Loop over pairs and compute the weighted circular mean phase over time (for each frequency)
    for p, pair in enumerate(pairs):

        # Extract phase and coherence for this pair in the frequency band of interest
        phase = phase_matrix[pair][freq_mask]  # shape (F, T)
        coherence = coherograms[pair][freq_mask]  # shape (F, T)

        # Weighted complex sum across time for each frequency
        weighted_sum = np.sum(coherence * np.exp(1j * phase), axis=1)  # (F,)
        # sum the coherence across time for each frequency
        weight_total = np.sum(coherence, axis=1)  # (F,)
        # Circular mean phase = angle of weighted average
        phi = np.angle(weighted_sum / weight_total)
        phi_obs[p, :] = phi  # store the observed phase for this pair

        # Also compute the ordinary (arithmetic) mean coherence per freq across time
        mean_coh = np.mean(coherence, axis=1)
        coh_avg[p, :] = mean_coh  # store the mean coherence

    # Build final weights for misfit: raise mean coherence to the 2nd power to emphasize high-coherence frequency bins (tunable).
    coh_weight = coh_avg ** 2
    coh_weight[coh_avg < coh_threshold] = 0.0  # apply threshold: If a frequency has little coherence, we don't even want to trust the circular mean phase.

    return pairs, phi_obs, coh_weight

def find_acoustic_source(phi_obs, coh_weight, data, pairs, codes,
                                station_coords, center_lat, center_lon, grid_spacing_m,
                                grid_width_km, grid_height_km, c_sound=340.0):
    """
    Searches through a lat/lon grid and computes theoretical inter-station phase distributions,
    and compares these with observations to find the best-fitting acoustic source location.
    """
    # We'll again need a frequency vector of interest for this
    freq_mask = (data.freq_vector >= data.freq_min) & (data.freq_vector <= data.freq_max)
    f_band = data.freq_vector[freq_mask]  # extract frequency vector of interest
    omega = 2.0 * np.pi * f_band  # angular frequency vector of interest

    # Convert specified grid spacing to degrees (w/ helper function)
    dlat, dlon = meters_to_degrees(grid_spacing_m, lat_deg=center_lat)

    # ---------------------------
    # Build lat/lon grid
    # ---------------------------

    # establish kilometer to degree conversions  at the center latitude
    km_to_lat, km_to_lon = meters_to_degrees(1000, lat_deg=center_lat)

    # Define latitude and longitude half-widths/heights in degrees
    lat_half_deg = (grid_height_km / 2.0) * km_to_lat
    lon_half_deg = (grid_width_km / 2.0) * km_to_lon

    # Construct arrays of latitudes and longitudes that cover the grid search box.
    lat_vec = np.arange(center_lat - lat_half_deg, center_lat + lat_half_deg + dlat, dlat)
    lon_vec = np.arange(center_lon - lon_half_deg, center_lon + lon_half_deg + dlon, dlon)

    # Grid dimensions: NX = number of longitudes, NY = number of latitudes.
    NX, NY = len(lon_vec), len(lat_vec)

    # Preallocate phase misfit array
    misfit = np.full((NX, NY), np.inf)

    # Precompute pair index arrays that map pair names to their respective station indices
    idx0 = np.array([codes.index(pair[0]) for pair in pairs], dtype=int) # index of first station in each pair
    idx1 = np.array([codes.index(pair[1]) for pair in pairs], dtype=int) # index of second station in each pair

    # Loop over grid cells and compute theoretical phase and misfit
    total_cells = NX * NY
    pbar = tqdm(total=total_cells, desc="Grid points")

    for i, lon in enumerate(lon_vec):
        for j, lat in enumerate(lat_vec):
            # get station coordinates as arrays
            station_lons = np.array([station_coords[code][0] for code in codes])
            station_lats = np.array([station_coords[code][1] for code in codes])

            # compute distances from current grid cell to each station (in meters)
            d_all = np.array([gps2dist_azimuth(lat, lon, sta_lat, sta_lon)[0] for sta_lat, sta_lon in zip(station_lats, station_lons)])

            # Distances for station pairs
            d1 = d_all[idx0]
            d2 = d_all[idx1]

            # Travel-time difference between stations for each pair
            tt_1 = d1 / c_sound  # travel time from source to station 1 in pair
            tt_2 = d2 / c_sound  # travel time from source to station 2 in pair
            delta_t = tt_1 - tt_2

            # Theoretical phase (P,F), wrapped to [-pi, pi] using delta_t
            phi_theo = np.angle(np.exp(1j * (omega[None, :] * delta_t[:, None])))

            # Residual between observed and theoretical phase (P,F)
            res = np.angle(np.exp(1j * (phi_obs - phi_theo)))

            # Weighted RMS misfit using circular distance (this down-weights incoherent frequencies)
            rms = np.sqrt(np.sum(coh_weight * (1 - np.cos(res))) / np.sum(coh_weight))

            misfit[i, j] = rms # store misfit value for this grid point.
            pbar.update(1)

    pbar.close()

    # Find best grid cell and return location
    idx_best = np.unravel_index(np.nanargmin(misfit), misfit.shape)

    lon_best = lon_vec[idx_best[0]]
    lat_best = lat_vec[idx_best[1]]

    return lat_best, lon_best, misfit, lon_vec, lat_vec, idx_best
