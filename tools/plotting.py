import numpy as np
from obspy.geodetics import gps2dist_azimuth
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as dates
#mpl.use('Qt5Agg')

def plot_network_coherence(Cxy2_norm, data, save_dir, save=False):
    # set colormap and colorbar limits
    colorm = LinearSegmentedColormap.from_list('', ['white', *plt.get_cmap('magma_r').colors])
    c_lim = [0.4, 1.0]

    if data.n_weeks > 1: # A single plotting row should not contain more than 7 days of data
        n_rows = int(np.ceil(data.n_weeks))
    else:
        n_rows = 1

    # Set up time bounds for the rows if > 1 row
    start_time = data.starttime.matplotlib_date
    end_time = data.endtime.matplotlib_date
    duration = end_time - start_time
    row_bounds = [(start_time + i * duration / n_rows, start_time + (i + 1) * duration / n_rows) for i in range(n_rows)]

    fig, axs = plt.subplots(n_rows, 1, figsize=(16, 10), sharex=False, sharey=True)
    plt.subplots_adjust(left=0.06, right=0.94, bottom=0.06, top=0.88, hspace=0.15)  # adjust spacing as needed

    if n_rows == 1:  # Ensure axs is a list for looping
        axs = [axs]

    for idx, ax in enumerate(axs):

        row_start, row_end = row_bounds[idx]  # start and end times for this row

        # Create a mask for indices corresponding to this week
        mask = (data.t >= row_start) & (data.t < row_end)
        row_t = data.t[mask]  # time vector for this week
        row_Cxy2_norm = Cxy2_norm[:, mask]  # coherence matrix for this week

        # Plot the median network coherence for this row
        mesh = ax.imshow(row_Cxy2_norm, aspect='auto', cmap=colorm, origin='lower',
                         extent=[row_start, row_end, data.freq_vector[0],
                                 data.freq_vector[-1]], interpolation='none')
        mesh.set_clim(c_lim)
        ax.set_ylim(data.freq_min, data.freq_max)
        ax.set_xlim(row_start, row_end)
        ax.set_ylabel('Frequency [Hz]')

        ax.xaxis_date()
        if n_rows == 1:
            ax.xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d %H'))
        else:
            ax.xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d'))
        ax.tick_params(axis='x')#, labelsize=10)

    # Label the x-axis on the bottom subplot only
    axs[-1].set_xlabel('UTC Time')#,fontsize=14)

    # add colorbar
    cbar = fig.colorbar(mesh, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label("Median Network\nMag$^2$ Coherence")

    fig.suptitle(
        f"{data.source_name} Median Network Coherence\n{data.nPairs} unique station pairs used,"
        f" {data.channel_str} channel\nStations: {data.station_names}", y=0.95)

    if save:
        plt.savefig(f"{save_dir}.jpg", dpi=300)

    plt.show()

    return fig, axs


def plot_interstation_coherence(coherograms, st, data, save_dir, save=False):

    station_distances = []
    for tr in st:
        # find distance from source to each station
        dist_m, _, _ = gps2dist_azimuth(data.source_lat, data.source_lon, tr.stats.latitude, tr.stats.longitude)
        station_distances.append((tr.stats.station, dist_m))  # store station name and distance

    stations_sorted_asc = sorted(station_distances, key=lambda x: x[1])  # sort stations by distance

    N = len(stations_sorted_asc)

    colorm = LinearSegmentedColormap.from_list('', ['white', *plt.get_cmap('magma_r').colors])
    c_lim = [0.4, 1.0]

    fig, axs = plt.subplots(N - 1, N - 1, figsize=(14, 12))
    plt.subplots_adjust(left=0.07, bottom=0.07, hspace=0.1, top=0.95, wspace=0.1)
    plt.subplots_adjust(left=0.07, bottom=0.07, hspace=0.1, top=0.95, wspace=0.1)

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
            ax.set_xticks([])



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

    plt.suptitle(f"{data.source_name} inter-station coherence contributions"
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


