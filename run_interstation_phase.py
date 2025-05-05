import os
from obspy.core import UTCDateTime, Stream
from obspy.clients.fdsn import Client
from waveform_collection import gather_waveforms_bulk
from obspy.geodetics import kilometer2degrees
from tools import toolbox, plotting
#%% FREQUENCY BAND AND WINDOW SETTINGS----------------------------------------------------------------------------------
FREQ_MIN = 0.1  # [Hz]
FREQ_MAX = 10
# Window length [sec]
WINDOW_LENGTH = 2*60
# Fraction of window overlap [0.0, 1.0)
WINDOW_OVERLAP = 0.75  # 0.75

# TIME, CHANNEL, AND SOURCE INFORMATION---------------------------------------------------------------------------------
STARTTIME = UTCDateTime("2025-04-27T13:30:00")
ANALYSIS_LEN = 30 * 60  # Length of analysis [sec]
ENDTIME = STARTTIME + ANALYSIS_LEN

CHANNEL = "*HZ"  # select "*HZ" or "*HN, *HE" for vertical or horizontals, respectively

SOURCE_NAME = "Akutan_Tectonic"
SOURCE_LAT = 54.134
SOURCE_LON = -165.986

MAX_RADIUS = 15  # max radius to search for stations [km]
STATIONS_TO_REMOVE = ["BRPK", "BKG", "N20K","SPCN", "RDE", "RED"]  # Remove these stations from the analysis

FILENAME = f"interstation_phase_{SOURCE_NAME}_{STARTTIME.year}_{STARTTIME.month}_{STARTTIME.day}"
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
st = toolbox.remove_network_response(st, inv, type='full')  # other option 'full'

if st[0].stats.channel[1:] == 'HE' or st[0].stats.channel[1:] == 'HN':
    st = toolbox.rotate_stations(st, inv, SOURCE_LAT, SOURCE_LON, output='radial')  # rotate to radial or transverse

#%% Get phase between all station pairs
phase_matrix, data, nPairs = toolbox.get_interstation_phase(st, window_length=WINDOW_LENGTH, window_overlap=WINDOW_OVERLAP, n_jobs=6)

data.add_plotting_info(FREQ_MIN, FREQ_MAX, CHANNEL, STARTTIME, ENDTIME, nPairs, SOURCE_NAME, SOURCE_LAT, SOURCE_LON)
# ----------------------------------------------------------------------------------------------------------------------
#%% Retrieve all station-pair coherograms in parallel
coherograms = toolbox.get_interstation_coherograms(st, data, n_jobs=6)

#%% PLOTTING Station-pair coherence contributions (Meant for short periods of data, ~1 day to a few)
fig, axs = plotting.plot_interstation_phase(phase_matrix, coherograms, st, data, save_dir=SAVE_DIR, coh_mask=True, save=False)