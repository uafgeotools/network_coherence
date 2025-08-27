import os
from obspy.core import UTCDateTime, Stream
from obspy.clients.fdsn import Client
from waveform_collection import gather_waveforms_bulk
from obspy.geodetics import kilometer2degrees
from tools import toolbox, plotting
import time
#%% FREQUENCY BAND AND WINDOW SETTINGS----------------------------------------------------------------------------------
FREQ_MIN = 0.1  # [Hz]
FREQ_MAX = 10

DOWNSAMPLE = True  # Downsample data for faster processing

# Window length [sec]
WINDOW_LENGTH = 1*60*60
# Fraction of window overlap [0.0, 1.0)
WINDOW_OVERLAP = 0.25  # 0.5

# TIME, CHANNEL, AND SOURCE INFORMATION---------------------------------------------------------------------------------
STARTTIME = UTCDateTime("2025-01-01T00:00:00")
ANALYSIS_LEN = 238 * 24 * 60 * 60  # Length of analysis [sec] # 237
ENDTIME = STARTTIME + ANALYSIS_LEN

CHANNEL = "*HZ"  # select "*HZ" or "*HN, *HE" for vertical or horizontals, respectively

SOURCE_NAME = "Shishaldin Volcano"
SOURCE_LAT = 54.7554  # Source latitude
SOURCE_LON = -163.9711  # Source longitude

#SOURCE_NAME = "Klyuchevskoy Volcanic Group"
#SOURCE_LAT = 56.056
#SOURCE_LON = 160.642
#SOURCE_LAT = 61.264770025202544  # Source latitude (Crater Peak)
#SOURCE_LON = -152.239314483138  # Source longitude (Crater Peak)

MAX_RADIUS = 18  # max radius to search for stations [km]
STATIONS_TO_REMOVE = ["N20K"]  # Remove these stations from the analysis

FILENAME = f"network_coherence_{SOURCE_NAME}_{STARTTIME.year}_{STARTTIME.month}_{STARTTIME.day}"
SAVE_DIR = f"{os.getcwd()}/figures/{FILENAME}"
# ----------------------------------------------------------------------------------------------------------------------
#%% Get obspy stream and inventory for stations within our search
client = Client("IRIS")
inv = client.get_stations(latitude=SOURCE_LAT, longitude=SOURCE_LON,
                          maxradius=kilometer2degrees(MAX_RADIUS),
                          network="*", station="*", location="*",
                          channel=CHANNEL, starttime=STARTTIME,
                          endtime=ENDTIME, level="response")

st = gather_waveforms_bulk(lon_0=SOURCE_LON, lat_0=SOURCE_LAT, max_radius=MAX_RADIUS, network='*',
                           station='*', location='*', channel=CHANNEL, starttime=STARTTIME, endtime=ENDTIME, n_jobs=6)

# Filter out bad stations from st and inv
for remove in STATIONS_TO_REMOVE:
    st = Stream(tr for tr in st if tr.stats.station != remove)
    for network in inv:
        network.stations = [station for station in network.stations if station.code != remove]


#%% Remove response, interpolate (if needed), and rotate if horizontals
st = toolbox.remove_network_response(st, inv, type='sensitivity')  # other option 'full'

if st[0].stats.channel[1:] == 'HE' or st[0].stats.channel[1:] == 'HN':
    st = toolbox.rotate_stations(st, inv, SOURCE_LAT, SOURCE_LON, output='radial')  # rotate to radial or transverse

if DOWNSAMPLE:  # Decimate to improve processing speed
    if (2 * FREQ_MAX) + 2 < st[0].stats.sampling_rate:
        decimation_factor = 2
    else:
        print('Data not downsampled, Original Nyquist too low for desired FREQ_MAX')
        decimation_factor = 1
    st.decimate(factor=decimation_factor)
print(st)
#%% Get median network coherence in parallel
start = time.time()
Cxy2_norm, data = toolbox.get_network_coherence(st, WINDOW_LENGTH, WINDOW_OVERLAP, n_jobs=6)
# Add plotting info to data class2
data.add_plotting_info(FREQ_MIN, FREQ_MAX, CHANNEL, STARTTIME, ENDTIME, SOURCE_NAME, SOURCE_LAT, SOURCE_LON)
end = time.time()
print(f"Coherence calculation took {end - start:.2f} seconds")
#%% PLOTTING Median network coherogram
fig, axs = plotting.plot_network_coherence(Cxy2_norm, data, save_dir=SAVE_DIR, save=False)
# ----------------------------------------------------------------------------------------------------------------------
#%% Retrieve all station-pair coherograms in parallel
# (separate from network coherence, requires a different parallelization, and requires 'data' from above.)
coherograms = toolbox.get_interstation_coherograms(st, data, n_jobs=6)

#%% PLOTTING Station-pair coherence contributions (Meant for short periods of data, ~1 day to a few)
SAVE_DIR = f"{os.getcwd()}/figures/interstation_coherence_{SOURCE_NAME}_{STARTTIME.year}_{STARTTIME.month}_{STARTTIME.day}"
fig, axs = plotting.plot_interstation_coherence(coherograms, st, data, save_dir=SAVE_DIR, save=False)
