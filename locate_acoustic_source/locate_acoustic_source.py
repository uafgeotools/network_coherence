import os
from obspy.core import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import kilometer2degrees
from waveform_collection import gather_waveforms_bulk
from tools import toolbox
from locate_acoustic_source.locate_tools import location_tools
from locate_acoustic_source.locate_tools import plotting
#%% USER INPUTS ----------------------------------------------------------------------------------
# Frequency band (pick this based off your coherent signal)
FREQ_MIN = 6   # [Hz]
FREQ_MAX = 7

# Window length [sec]
WINDOW_LENGTH = 10*60  # 6 mins before
WINDOW_OVERLAP = 0.75  # fraction of overlap

# Time, channel, source info
STARTTIME = UTCDateTime("2025-09-14T21:45:00")
ANALYSIS_LEN = 24 * 60 * 60  # [sec]
ENDTIME = STARTTIME + ANALYSIS_LEN

CHANNEL = "*DF"

SOURCE_NAME = "I53US"
CENTER_LAT = 64.866166  # grid center latitude
CENTER_LON = -147.85673  # grid center longitude

MAX_RADIUS = 3  # km (max radius from grid-center to search for stations)
STATIONS_TO_REMOVE = []

FILENAME = f"locate_acoustic_source_{SOURCE_NAME}_{STARTTIME.year}_{STARTTIME.month}_{STARTTIME.day}"
SAVE_DIR = f"{os.getcwd()}/figures/{FILENAME}"

# Grid search parameters
c = 336.6  # acoustic velocity (m/s) assuming horizontal propagation.
grid_spacing_m = 20.0  # grid spacing in meters
grid_width_km = 4.0     # grid x width in km
grid_height_km = 4.0    # grid y width in km
#%% Get waveforms, remove response, and compute interstation phase/coherency --------------
client = Client("IRIS")
inv = client.get_stations(latitude=CENTER_LAT, longitude=CENTER_LON,
                          maxradius=kilometer2degrees(MAX_RADIUS),
                          network="*", station="*", location="*",
                          channel=CHANNEL, starttime=STARTTIME,
                          endtime=ENDTIME, level="response")

st = gather_waveforms_bulk(lon_0=CENTER_LON, lat_0=CENTER_LON, max_radius=MAX_RADIUS,
                           network='*', station='*', location='*', channel=CHANNEL,
                           starttime=STARTTIME, endtime=ENDTIME, n_jobs=6)

st = toolbox.remove_network_response(st, inv, type='full') # remove instrument response

# compute time-frequency inter-station phases
phase_matrix, data, nPairs = toolbox.get_interstation_phase(st, window_length=WINDOW_LENGTH,
                                                           window_overlap=WINDOW_OVERLAP, n_jobs=6)
# add some important stuff to the data class
data.add_plotting_info(FREQ_MIN, FREQ_MAX, CHANNEL, STARTTIME, ENDTIME,
                       SOURCE_NAME, CENTER_LAT, CENTER_LON)

# get inter-station  time-frequency coherograms
coherograms = toolbox.get_interstation_coherograms(st, data, n_jobs=6)

#%%
# extract station lat/lon
codes, station_coords, station_lats, station_lons = [], {}, [], []
for net in inv:
    for sta in net:
        codes.append(sta.code)
        station_coords[sta.code] = (sta.longitude, sta.latitude)
        station_lons.append(sta.longitude)
        station_lats.append(sta.latitude)

# Compute observed inter-station phases (using coherence weighting) and determine final coherence weighting for location algorithm
pairs, phi_obs, coh_weight = location_tools.compute_phi_obs_and_weights(phase_matrix, coherograms, data, coh_threshold=0.6)
#%%
# Run lat/lon grid search
lat_best, lon_best, misfit_grid, lon_vec, lat_vec, idx_best = location_tools.find_acoustic_source(phi_obs, coh_weight, data, pairs, codes,
                                station_coords, CENTER_LAT, CENTER_LON, grid_spacing_m,
                                grid_width_km, grid_height_km, c_sound=c)

print(f"Best 2D acoustic source: lat={lat_best:.5f}, lon={lon_best:.5f}")

#%%
fig, ax = plotting.plot_misfit_latlon(misfit_grid, lon_vec, lat_vec, idx_best, station_coords, SAVE_DIR, save=False)
