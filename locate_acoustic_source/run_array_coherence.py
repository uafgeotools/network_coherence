import os
from obspy.core import UTCDateTime
from obspy.clients.fdsn import Client
from waveform_collection import gather_waveforms
from tools import toolbox, plotting, array_plotting
#%% FREQUENCY BAND AND WINDOW SETTINGS----------------------------------------------------------------------------------
FREQ_MIN = 0.01  # [Hz]
FREQ_MAX = 9
# Window length [sec]
WINDOW_LENGTH = 6*60
# Fraction of window overlap [0.0, 1.0)
WINDOW_OVERLAP = 0.75  # 0.75

# TIME, CHANNEL, AND SOURCE INFORMATION---------------------------------------------------------------------------------
STARTTIME = UTCDateTime("2025-09-14T21:45:00")
ANALYSIS_LEN = 3 * 60 * 60  # Length of analysis [sec]
ENDTIME = STARTTIME + ANALYSIS_LEN

NETWORK = "IM"
STATION = "I53*"
CHANNEL = "*DF"

FILENAME = f"array_coherence_{STATION}_{STARTTIME.year}_{STARTTIME.month}_{STARTTIME.day}"
SAVE_DIR = f"{os.getcwd()}/figures/array/{FILENAME}"
# ----------------------------------------------------------------------------------------------------------------------
#%% Get obspy stream and inventory for stations within our search
client = Client("IRIS")
inv = client.get_stations(network=NETWORK, station=STATION, location="*",
                          channel=CHANNEL, starttime=STARTTIME,
                          endtime=ENDTIME, level="response")

st = gather_waveforms(source="IRIS",network=NETWORK,station=STATION, location='*', channel=CHANNEL, starttime=STARTTIME, endtime=ENDTIME, n_jobs=6)


#%% Remove response, interpolate (if needed), and rotate if horizontals
st = toolbox.remove_network_response(st, inv, type='sensitivity')  # 'full', or 'sensitivity'

#%% Get median network coherence in parallel
import time
start = time.time()
Cxy2_norm, data = toolbox.get_network_coherence(st, WINDOW_LENGTH, WINDOW_OVERLAP, n_jobs=6)
# Add plotting info to data class
data.add_plotting_info(FREQ_MIN, FREQ_MAX, CHANNEL, STARTTIME, ENDTIME,
                       source_name=STATION,source_lat=None, source_lon=None)
end = time.time()
print(f"Coherence calculation took {end - start:.2f} seconds")
#%% PLOTTING Median network coherogram
fig, axs = plotting.plot_network_coherence(Cxy2_norm, data, save_dir=SAVE_DIR, save=False)

# ----------------------------------------------------------------------------------------------------------------------
#%% Retrieve all array-pair coherograms in parallel

#%%
# (separate from network coherence, requires a different parallelization, and requires 'data' from above.)
coherograms = toolbox.get_interstation_coherograms(st, data, n_jobs=6)

#%% PLOTTING array-pair coherence contributions (Meant for short periods of data, ~1 day to a few)
SAVE_DIR = f"{os.getcwd()}/figures/array/interelement_coherence_{STATION}_{STARTTIME.year}_{STARTTIME.month}_{STARTTIME.day}"
fig, axs = plotting.plot_interstation_coherence(coherograms, st, data, save_dir=SAVE_DIR, save=False)

#%% Compare network array coherence to array processing results:
import numpy as np
from obspy.geodetics import gps2dist_azimuth
SAVE_DIR = f"{os.getcwd()}/figures/array/array_proccesing_v_coherence_{STATION}_{STARTTIME.year}_{STARTTIME.month}_{STARTTIME.day}_{FREQ_MIN}_{FREQ_MAX}"
fig, axs = array_plotting.plot_array_v_coherence(st, Cxy2_norm, data, ALPHA_LTS=0.75, save_dir=SAVE_DIR, save=False)
latlist = [tr.stats.latitude for tr in st]
lonlist = [tr.stats.longitude for tr in st]
array_lat = np.mean(latlist)
array_lon = np.mean(lonlist)
#erebus_lat = -77.528042
#erebus_lon = 167.16072
#_, _, baz = gps2dist_azimuth(erebus_lat, erebus_lon, array_lat, array_lon)
#axs[3].axhline(baz, color='k', linestyle='--', lw=1, label='Mount Erebus')
#axs[3].legend()
fig.savefig(f"{SAVE_DIR}.jpg",dpi=300)
fig.show()
#%%

#%%
