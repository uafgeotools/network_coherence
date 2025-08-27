import numpy as np
from obspy.geodetics import gps2dist_azimuth
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable
import matplotlib.dates as dates
from matplotlib import rcParams
import matplotlib as mpl
mpl.use('Qt5Agg')
rcParams.update({'font.size': 14,'axes.labelsize': 16, 'axes.titlesize': 14,})

import matplotlib.pyplot as plt
import matplotlib.dates as dates
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap

def plot_network_coherence(Cxy2_norm, data, save_dir, save=False, cmin=0.4, cmax=1):
    # --- colormap for coherence ---
    colorm = LinearSegmentedColormap.from_list(
        '', ['white', *plt.get_cmap('magma_r').colors]
    )
    c_lim = [cmin, cmax]

    # --- figure and axis ---
    fig, ax = plt.subplots(figsize=(16, 10))
    plt.subplots_adjust(left=0.06, right=0.94, bottom=0.1, top=0.86)

    # --- main coherence image ---
    mesh = ax.imshow(
        Cxy2_norm,
        aspect='auto',
        cmap=colorm,
        origin='lower',
        extent=[data.starttime.matplotlib_date,
                data.endtime.matplotlib_date,
                data.freq_vector[0],
                data.freq_vector[-1]],
        interpolation='none'
    )
    mesh.set_clim(c_lim)
    ax.set_ylim(data.freq_min, data.freq_max)
    ax.set_xlim(data.starttime.matplotlib_date, data.endtime.matplotlib_date)
    ax.set_ylabel('Frequency [Hz]')

    # x-axis formatting
    x_ticks = np.linspace(data.starttime.matplotlib_date, data.endtime.matplotlib_date, 6)
    ax.set_xticks(x_ticks)
    ax.xaxis_date()
    if data.n_days <= 1:
        ax.xaxis.set_major_formatter(dates.DateFormatter('%m-%d %H:%M'))
    else:
        ax.xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d %H'))
    ax.tick_params(axis='x')
    ax.grid(alpha=0.8)

    # --- median strip (right side) ---
    median_strip = get_median_strip(Cxy2_norm, data, fbin=0.1)
    strip_ax = ax.inset_axes([1.0, 0, 0.03, 1], transform=ax.transAxes)
    median_mesh = strip_ax.imshow(
        median_strip,
        aspect='auto',
        cmap=colorm,
        origin='lower',
        interpolation='none'
    )
    median_mesh.set_clim(c_lim)
    strip_ax.set_xticks([])
    strip_ax.set_yticks([])
    strip_ax.text(1, 1, 'Median', rotation=45, ha='center', va='bottom',
                  transform=strip_ax.transAxes)

    # --- greyscale station counts strip (on top) ---
    n_stations_total = int(getattr(data, 'nStations', 0) or 0)
    values = [0] + list(range(2, n_stations_total + 1))  # skip 1
    base_cmap = plt.get_cmap('Greys_r')
    colors = base_cmap(np.linspace(0.0, 1.0, len(values)))
    greys_cmap = ListedColormap(colors)
    grey_norm = BoundaryNorm(np.arange(len(values) + 1) - 0.5,
                             ncolors=len(values), clip=True)

    n_stations = np.asarray(data.n_station_contributions, dtype=int)
    mapped = np.full_like(n_stations, fill_value=-1, dtype=int)
    for i, v in enumerate(values):
        mapped[n_stations == v] = i

    pair_ax = ax.inset_axes([0, 1, 1, 0.02], transform=ax.transAxes)
    pair_ax.imshow(mapped[np.newaxis, :],
                   aspect='auto',
                   cmap=greys_cmap, norm=grey_norm,
                   extent=[data.starttime.matplotlib_date,
                           data.endtime.matplotlib_date, 0, 1],
                   origin='lower', interpolation='nearest')
    pair_ax.set_yticks([])
    pair_ax.set_xticks([])

    # --- labels ---
    ax.set_xlabel('UTC Time')

    # --- coherence colorbar ---
    cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', fraction=0.05, pad=0.05)
    cbar.set_label(f"Median {data.label} Mag$^2$ Coherence")

    # --- station counts colorbar ---
    sm = ScalarMappable(cmap=greys_cmap, norm=grey_norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.03, 0.94, 0.2, 0.015])  # left, top, width, height
    cbar2 = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar2.set_ticks(np.arange(len(values)))
    cbar2.set_ticklabels([str(v) for v in values])
    cbar2.set_label(f'# of Contributing {data.sub_label}')

    # --- title ---
    fig.suptitle(
        f"{data.source_name} Median {data.label} Coherence, {data.channel_str} channel"
        f"\n{data.sub_label}: {data.station_names}", y=0.95
    )

    if save:
        plt.savefig(f"{save_dir}.jpg", dpi=300)

    plt.show()
    return fig, ax



def plot_interstation_coherence(coherograms, st, data, save_dir, save=False):
    station_distances = []
    for tr in st:
        if data.source_lat == None and data.source_lon == None:  # Do this for arrays so we sort by distance from element 1
            data.source_lat = tr.stats.latitude
            data.source_lon = tr.stats.longitude
            data.source_name = tr.stats.station
        # find distance from source to each station
        dist_m, _, _ = gps2dist_azimuth(data.source_lat, data.source_lon, tr.stats.latitude, tr.stats.longitude)
        station_distances.append((tr.stats.station, dist_m))  # store station name and distance

    stations_sorted_asc = sorted(station_distances, key=lambda x: x[1])  # sort stations by distance

    N = len(stations_sorted_asc)

    colorm = LinearSegmentedColormap.from_list('', ['white', *plt.get_cmap('magma_r').colors])
    c_lim = [0.4, 1.0]

    fig, axs = plt.subplots(N - 1, N - 1, figsize=(14, 12))
    plt.subplots_adjust(left=0.08, bottom=0.09, hspace=0.1, top=0.95, wspace=0.1)

    ct = 0
    for i in range(0, N - 1):
        for j in range(0, N - 1):
            ax = axs[i, j]  # select correct axis
            station_y = stations_sorted_asc[N - 2 - i][0]
            dist_y = stations_sorted_asc[N - 2 - i][1]

            station_x = stations_sorted_asc[j + 1][0]
            dist_x = stations_sorted_asc[j + 1][1]

            if not (N - 2 - i < j + 1):  # Turn off the panels in the upper left triangular
                ax.set_frame_on(False)
                ax.axis('off')
                continue

            # find the coherogram for this particular station pair
            if (station_y, station_x) in coherograms:
                coherogram = coherograms[(station_y, station_x)]
            elif (station_x, station_y) in coherograms:
                coherogram = coherograms[(station_x, station_y)]


            coherogram_with_strip, extended_t = add_median_strip(coherogram, 
                                                                 data,
                                                                 fbin=0.5, 
                                                                 thickness=15)

            # now plot
            mesh = ax.imshow(coherogram_with_strip, aspect='auto', cmap=colorm,
                             origin='lower',
                             extent=[extended_t[0], extended_t[-1],
                                     data.freq_vector[0],
                                     data.freq_vector[-1]],
                             interpolation='none')

            mesh.set_clim(c_lim)
            ax.set_ylim(data.freq_min, data.freq_max)

            # label the lowermost left panel only
            if i == N - 2 and j == 0:
                print("Adding median strip label")
                ax.text(1, 1.03, 'Median', color='black', fontsize=10,
                        ha='center', va='center', rotation=0,
                        transform=ax.transAxes)

            # add time ticks
            ax.xaxis_date()
            ax.set_xticklabels([])

            if i == (N - 2) - j:  # set y-axis station labels on the filled panels on the left
                ax.set_ylabel(f"$\\bf{{{station_y}}}$\n {np.round(dist_y/1000,decimals=1)} km")
            else:
                ax.set_yticks([])

            if i == N - 2:  # set x-axis labels for bottom row only
                ax.set_xlabel(f"$\\bf{{{station_x}}}$\n {np.round(dist_x/1000, decimals=1)} km")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(mesh, cax=cbar_ax)  # add colorbar
    cbar.set_label("Mag$^2$ Coherence", fontsize=16)

    # add overarching x-axis label
    fig.text(0.5, 0.02, f'Increasing distance from {data.source_name} --------------->',
             ha='center', va='center', fontsize=16)
    # add overarching y-axis label
    fig.text(0.30, 0.55, f'Increasing distance from {data.source_name} --------------->',
             ha='center', va='center', fontsize=16, rotation=45)

    plt.suptitle(f"{data.source_name} inter-{data.sub_label[:-1]} coherence contributions"
                 f"\n{data.starttime.year}-{data.starttime.month}-{data.starttime.day} to {data.endtime.month}-{data.endtime.day}",
                 fontsize=18)

    if save:
        plt.savefig(f"{save_dir}.jpg", dpi=300)

    plt.show()

    return fig, axs


def plot_interstation_phase(phase_matrix, coherograms, st, data, save_dir, coh_mask=True, save=False):

    station_distances = []
    for tr in st:
        # find distance from source to each station
        dist_m, _, _ = gps2dist_azimuth(data.source_lat, data.source_lon, tr.stats.latitude, tr.stats.longitude)
        station_distances.append((tr.stats.station, dist_m))  # store station name and distance

    stations_sorted_asc = sorted(station_distances, key=lambda x: x[1])  # sort stations by distance

    N = len(stations_sorted_asc)

    colorm = 'twilight_shifted'  # colormap for phase

    fig, axs = plt.subplots(N - 1, N - 1, figsize=(14, 12))
    plt.subplots_adjust(left=0.09, bottom=0.09, hspace=0.1, top=0.95, wspace=0.1)

    for i in range(0, N - 1):
        for j in range(0, N - 1):
            ax = axs[i, j]  # select correct axis
            station_y = stations_sorted_asc[N - 2 - i][0]
            dist_y = stations_sorted_asc[N - 2 - i][1]

            station_x = stations_sorted_asc[j + 1][0]
            dist_x = stations_sorted_asc[j + 1][1]

            if not (N - 2 - i < j + 1):  # Turn off the panels in the upper left triangular
                ax.set_frame_on(False)
                ax.axis('off')
                continue

            # find the coherogram for this particular station pair
            if (station_y, station_x) in coherograms:
                coherogram = coherograms[(station_y, station_x)]
                phase_mat = phase_matrix[(station_y, station_x)]

            elif (station_x, station_y) in coherograms:
                coherogram = coherograms[(station_x, station_y)]
                phase_mat = phase_matrix[(station_x, station_y)]

            # now plot
            if coh_mask:
                coherogram = coherogram = np.clip(coherogram, 0, 1)
                mesh = ax.imshow(phase_mat, aspect='auto', cmap=colorm,
                                 origin='lower',
                                 extent=[data.t[0], data.t[-1],
                                         data.freq_vector[0],
                                         data.freq_vector[-1]],
                                 interpolation='none', alpha=coherogram)
            else:
                mesh = ax.imshow(phase_mat, aspect='auto', cmap=colorm,
                                 origin='lower',
                                 extent=[data.t[0], data.t[-1],
                                         data.freq_vector[0],
                                         data.freq_vector[-1]],
                                 interpolation='none')

            ax.set_ylim(data.freq_min, data.freq_max)


            # add time ticks
            ax.xaxis_date()
            ax.set_xticklabels([])

            if i == (N - 2) - j:  # set y-axis station labels on the filled panels on the left
                ax.set_ylabel(f"$\\bf{{{station_y}}}$\n {np.round(dist_y/1000,decimals=1)} km")
            else:
                ax.set_yticks([])

            if i == N - 2:  # set x-axis labels for bottom row only
                ax.set_xlabel(f"$\\bf{{{station_x}}}$\n {np.round(dist_x/1000, decimals=1)} km")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(mesh, cax=cbar_ax)  # add colorbar
    cbar.set_label("Phase [radians]", fontsize=16)
    cbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cbar.set_ticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$'], fontsize=16)

    # add overarching x-axis label
    fig.text(0.5, 0.02, f'Increasing distance from {data.source_name} --------------->',
             ha='center', va='center', fontsize=16)
    # add overarching y-axis label
    fig.text(0.30, 0.55, f'Increasing distance from {data.source_name} --------------->',
             ha='center', va='center', fontsize=16, rotation=45)

    plt.suptitle(f"{data.source_name} inter-station phase"
                 f"\n{data.starttime.year}-{data.starttime.month}-{data.starttime.day} to {data.endtime.month}-{data.endtime.day}",
                 fontsize=18)

    if save:
        plt.savefig(f"{save_dir}.jpg", dpi=300)

    plt.show()

    return fig, axs


def add_median_strip(coherogram, data, fbin=0.5, thickness=15):
    """
    Add the median strip to the coherogram
    :param coherogram - 2D array of coherence values
    :param data: data object containing frequency vector and time vector
    :param fbin: frequency bin size, default is 0.5 Hz
    :param thickness: thickness of the median strip, default is 15 pixels
    :return: coherogram with median strip, extended time vector
    """

    # Create frequency z bins
    bin_edges = np.arange(data.freq_vector[0],
                          data.freq_vector[-1] + fbin, fbin)

    # Compute median coherence per bin
    bin_centers = []
    bin_medians = []

    for ii in range(len(bin_edges) - 1):
        f_start = bin_edges[ii]
        f_end = bin_edges[ii + 1]

        # Find frequency indices in this bin
        bin_indices = np.where((data.freq_vector >= f_start) & (
                data.freq_vector < f_end))[0]

        if len(bin_indices) > 0:
            subset = coherogram[bin_indices, :]
            median_val = np.median(subset)
            bin_centers.append((f_start + f_end) / 2)
            bin_medians.append(median_val)

    bin_centers = np.array(bin_centers)
    bin_medians = np.array(bin_medians)

    # Create thicker median strip (e.g., 10 pixels wide). Might need
    # to adjust this based on the size of the figure?
    median_strip = np.full((len(data.freq_vector), thickness), np.nan)

    # Fill each 0.5 Hz bin row with its median value across the width
    for ii in range(len(bin_edges) - 1):
        f_start = bin_edges[ii]
        f_end = bin_edges[ii + 1]

        bin_indices = np.where((data.freq_vector >= f_start) & (
                data.freq_vector < f_end))[0]

        if len(bin_indices) > 0:
            median_strip[bin_indices, :] = bin_medians[ii]

    # Combine with original coherogram
    coherogram_with_strip = np.hstack([coherogram, median_strip])

    # Extend time axis to reflect new column
    dt = data.t[1] - data.t[0]
    extended_t = np.append(data.t, data.t[-1] + dt)

    return coherogram_with_strip, extended_t


def get_median_strip(coherogram, data, fbin=0.5):
    # Compute median along each row
    medians = np.median(coherogram, axis=1)

    # Determine the desired frequency bins based on the specified fbin size
    min_freq = data.freq_min
    max_freq = data.freq_max
    desired_freqs = np.arange(min_freq, max_freq + fbin, fbin)

    # Find indices for new frequency discretization
    indices = []
    for f in desired_freqs:
        if f > max_freq:
            break
        idx = np.argmin(np.abs(data.freq_vector - f))
        if idx not in indices:  # Avoid duplicates
            indices.append(idx)

    # Return the medians at the selected indices
    return medians[indices].reshape(-1,1)