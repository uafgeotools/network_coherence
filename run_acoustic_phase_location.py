from obspy import UTCDateTime
from waveform_collection import gather_waveforms_bulk
from rtm import (define_grid, produce_dem)
from tools import toolbox, plotting

#%% Define
DEM_FILE = None  # optional path to DEM file for 3d distance calculations

LAT_0 = 64.866   # [deg] Latitude of grid center
LON_0 = -147.857  # [deg] Longitude of grid center

X_RADIUS = 2400  # [m] E-W grid radius (half of grid "width")
Y_RADIUS = 2400  # [m] N-S grid radius (half of grid "height")
SPACING = 2  # Grid spacing [m]

grid = define_grid(lon_0=LON_0, lat_0=LAT_0, x_radius=X_RADIUS,
                   y_radius=Y_RADIUS, spacing=SPACING, projected=True)

if DEM_FILE:
    dem = produce_dem(grid, external_file=DEM_FILE)
else:
    dem = None

#%% Gather DATA

# Start and end of time window containing event
STARTTIME = UTCDateTime("2025-06-13T11:00:00")
ANALYSIS_LEN = 4 * 60 * 60  # [sec]
ENDTIME = STARTTIME + ANALYSIS_LEN

# Windowing parameters and frequency band for inter-station phase calculation
WINDOW_LENGTH = 2*60  # [s]
WINDOW_OVERLAP = 0.75  # fraction of overlap

FREQ_MIN = 1.8  # [Hz] Lower cutoff frequency for phase calculation
FREQ_MAX = 2.6  # [Hz] Upper cutoff frequency for phase calculation

# Data collection parameters
NETWORK = '*'
STATION = '*'
LOCATION = '*'
CHANNEL = 'BDF'

MAX_RADIUS = 15  # [km] Max. radius from grid center to select stations

#%%
st = gather_waveforms_bulk(lon_0=LON_0, lat_0=LAT_0, max_radius=MAX_RADIUS,
                           network=NETWORK, station=STATION, location=LOCATION, channel=CHANNEL,
                           starttime=STARTTIME, endtime=ENDTIME, remove_response=True)
# Waveforms can also be gathered using "gather_waveforms" to select individual network/stations codes rather than radius search.

if len({tr.stats.sampling_rate for tr in st}) > 1:  # if more than one sample rate, interpolate to lowest fs
    rate = min(tr.stats.sampling_rate for tr in st)
    st.filter("lowpass", freq=rate / 2 - 2, corners=12, zerophase=True)
    st.interpolate(sampling_rate=rate, method="lanczos", a=15)

#%% Compute inter-station phase, coherence, and network coherence
phase_pairs, coherence_pairs, network_coherence, data = toolbox.get_interstation_phase_and_coherence(st, window_length=WINDOW_LENGTH,
                                                           window_overlap=WINDOW_OVERLAP, n_jobs=4) # change n_jobs to use additional/fewer CPU cores.
data.freq_min, data.freq_max = FREQ_MIN, FREQ_MAX
data.source_lat, data.source_lon = LAT_0, LON_0

# compute observed inter-station phases
phi_obs = toolbox.compute_phi_obs(phase_pairs, coherence_pairs, data)

# Plot coherent inter-station phase
fig_2, axs_2 = plotting.plot_interstation_phase(phase_pairs, coherence_pairs, phi_obs, st, data)

#%% Run grid search
acoustic_velocity = 335.75  # [m/s] change this based on local sound speed, or infer.

S = toolbox.grid_search_phase(st=st, S=grid, phase_obs=phi_obs,
                              wave_velocity=acoustic_velocity, dem=dem, data=data)

#%% Plot phase misfit grid (best-fit lat/lon will be plotted on figure)
fig_phase = plotting.plot_phase_grid(S, st, dem=dem, xy_grid=X_RADIUS, cont_int=25, annot_int=200) # contour and annotation intervals only show up if topographic DEM is present
