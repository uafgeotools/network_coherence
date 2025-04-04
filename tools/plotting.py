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
                         extent=[row_t[0], row_t[-1], data.freq_vector[0],
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
        ax.tick_params(axis='x', labelsize=10)

    # Label the x-axis on the bottom subplot only
    axs[-1].set_xlabel('UTC Time',fontsize=14)

    # add colorbar
    cbar = fig.colorbar(mesh, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label("Median Network\nMag$^2$ Coherence")

    fig.suptitle(
        f"{data.source_name} Median Network Coherence\n{data.nPairs} unique station pairs used,"
        f" {data.channel_str} channel\nStations: {data.station_names}", y=0.95)

    if save:
        plt.savefig(f"{save_dir}.jpg", dpi=300)
    end = time.time()
    print(f"Plotting time: {end - start:.2f} seconds")
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
    plt.subplots_adjust(left=0.07, bottom=0.05, hspace=0.05, top=0.95, wspace=0.04)

    for i in range(0, N - 1):
        for j in range(0, N - 1):
            ax = axs[i, j]  # select correct axis
            station_y = stations_sorted_asc[N - 2 - i][0]
            station_x = stations_sorted_asc[j + 1][0]

            if not (N - 2 - i < j + 1):  # Turn off the panels in the upper left triangular
                ax.set_frame_on(False)
                ax.axis('off')
                continue

            # find the coherogram for this particular station pair
            if (station_y, station_x) in coherograms:
                coherogram = coherograms[(station_y, station_x)]
            elif (station_x, station_y) in coherograms:
                coherogram = coherograms[(station_x, station_y)]

            # plot the coherogram
            mesh = ax.imshow(coherogram, aspect='auto', cmap=colorm, origin='lower',
                             extent=[data.t[0], data.t[-1], data.freq_vector[0],
                                      data.freq_vector[-1]], interpolation='none')
            mesh.set_clim(c_lim)
            ax.set_ylim(data.freq_min, data.freq_max)
            ax.set_xticks([])

            if i == (N - 2) - j:  # set y-axis station labels on the filled panels on the left
                ax.set_ylabel(station_y, fontweight='bold')
            else:
                ax.set_yticks([])

            if i == N - 2:  # set x-axis labels for bottom row only
                ax.set_xlabel(station_x, fontweight='bold')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(mesh, cax=cbar_ax)  # add colorbar
    cbar.set_label("Mag$^2$ Coherence")

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



