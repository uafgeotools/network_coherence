import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as dates
mpl.use('Qt5Agg')

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
        mesh = ax.pcolormesh(row_t, data.freq_vector, row_Cxy2_norm, shading='auto', cmap=colorm)
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
    plt.show()

    return fig, axs
