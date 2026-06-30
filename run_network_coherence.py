import os
from obspy.core import UTCDateTime, Stream
from obspy.clients.fdsn import Client
from waveform_collection import gather_waveforms_bulk
from obspy.geodetics import kilometer2degrees
from tools import toolbox, plotting
from scipy.signal import medfilt
#%% FREQUENCY BAND AND WINDOW SETTINGS----------------------------------------------------------------------------------
FREQ_MIN = 0.1  # [Hz]
FREQ_MAX = 10

# Window length [sec]
WINDOW_LENGTH = 2*60
# Fraction of window overlap [0.0, 1.0)
WINDOW_OVERLAP = 0.75  #

MED_FILT = True  # Apply median filter to coherograms

# TIME, CHANNEL, AND SOURCE INFORMATION---------------------------------------------------------------------------------
STARTTIME = UTCDateTime("2025-03-09T00:00:00")
ANALYSIS_LEN = 14 * 24 * 60 * 60  # [sec]
ENDTIME = STARTTIME + ANALYSIS_LEN

CHANNEL = "BHZ"  # select "*HZ" or "*HN, *HE" for vertical or horizontals, respectively

SOURCE_NAME = "Spurr"
SOURCE_LAT = 61.2989  # Source latitude (Spurr)
SOURCE_LON = -152.2539  # Source longitude (Spurr)

MAX_RADIUS = 25  # max radius to search for stations [km]
STATIONS_TO_REMOVE = ['N20K','SPCN']  # Remove these stations from the analysis
#-----------------------------------------------------------------------------------------------------------------------
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
print(st)

#%% Get median network coherence in parallel
Cxy2_net, data = toolbox.get_network_coherence(st, WINDOW_LENGTH, WINDOW_OVERLAP, n_jobs=6)
# Add plotting info to data class
data.add_plotting_info(FREQ_MIN, FREQ_MAX, CHANNEL, STARTTIME, ENDTIME, SOURCE_NAME, SOURCE_LAT, SOURCE_LON)

#%% Optional median filter smoothing of network coherogram
if MED_FILT:
    Cxy2_net= medfilt(Cxy2_net, kernel_size=(1, 5))

#%%  PLOTTING Median network coherogram
fig, axs = plotting.plot_network_coherence(Cxy2_net, data, cmin=0.35)

#%% Retrieve all station-pair coherograms in parallel
coherograms = toolbox.get_interstation_coherograms(st, data, n_jobs=6)

#%% median filter smoothed coherogram
if MED_FILT:
    for sta1, sta2 in coherograms:
        coherogram = coherograms[(sta1, sta2)]
        coherogram_smoothed = medfilt(coherogram, kernel_size=(1, 5))
        coherograms[(sta1, sta2)] = coherogram_smoothed

#%% PLOTTING Station-pair coherence contributions
fig, axs = plotting.plot_interstation_coherence(coherograms, st, data)
