import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as dates
from lts_array import ltsva
from matplotlib import rcParams
import numpy as np

rcParams.update({'font.size': 13, 'axes.labelsize': 12, 'axes.titlesize': 14, })


def plot_array_v_coherence(st, Cxy2_norm, data, ALPHA_LTS=0.5, save_dir=None, save=False):
    # Filter the stream
    st_filt = st.copy()
    st_filt.filter('bandpass', freqmin=data.freq_min, freqmax=data.freq_max, corners=2, zerophase=True)

    # Extract latitudes and longitudes
    latlist = [tr.stats.latitude for tr in st]
    lonlist = [tr.stats.longitude for tr in st]

    # Run ltsva to get array processing results
    vel_lts, baz_lts, t_lts, mdccm_lts, stdict_lts, sigma_tau, conf_int_vel, conf_int_baz = ltsva(
        st_filt, latlist, lonlist, data.window_length, data.window_overlap, ALPHA_LTS
    )

    # Create figure with four subplots, sharing x-axis, with first panel twice as tall
    fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True,
                            gridspec_kw={'height_ratios': [2, 1, 1, 1]},
                            constrained_layout=True)
    # Plot coherence (Cxy2_norm) in the top panel
    colorm = LinearSegmentedColormap.from_list('', ['white', *plt.get_cmap('magma_r').colors])

    mesh = axs[0].imshow(Cxy2_norm, aspect='auto', cmap=colorm, origin='lower',
                         extent=[data.t[0], data.t[-1], data.freq_vector[0], data.freq_vector[-1]],
                         interpolation='none')
    mesh.set_clim(0.4, 1.0)
    axs[0].set_ylim(data.freq_min, data.freq_max)
    axs[0].set_ylabel('Frequency [Hz]')
    axs[0].grid(alpha=0.8)

    # Plot waveform in the second panel (using the first trace)
    tr = st_filt[0]
    tvec = tr.times('matplotlib')
    axs[1].plot(tvec, tr.data, 'k')
    axs[1].set_ylabel('Amplitude [Pa]')
    axs[1].text(0.92, 0.90, tr.stats.station, transform=axs[1].transAxes)
    ax_right = axs[1].twinx()
    ax_right.set_ylabel(f"{data.freq_min}-{data.freq_max} Hz", fontsize=12, fontweight='bold', labelpad=20, rotation=270)
    ax_right.tick_params(right=False, labelright=False)

    # Plot trace velocity in the third panel, colored by mdccm_lts
    cm = 'RdYlBu_r'
    sc = axs[2].scatter(t_lts, vel_lts, c=mdccm_lts, cmap=cm, edgecolors='k', lw=0.3)
    axs[2].set_ylim(0.25, 0.45)
    axs[2].set_ylabel('Trace Velocity [km/s]')

    # Plot backazimuth in the fourth panel, colored by mdccm_lts
    sc = axs[3].scatter(t_lts, baz_lts, c=mdccm_lts, cmap=cm, edgecolors='k', lw=0.3)
    axs[3].set_ylim(0, 360)
    axs[3].set_ylabel('Backazimuth [deg]')
    axs[3].set_xlabel('UTC Time')
    axs[3].xaxis_date()
    axs[3].xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
    start_date = dates.num2date(data.t[0])
    text_string = start_date.strftime('%Y-%m')
    fig.text(0.02, 0.01, text_string, transform=fig.transFigure)

    x_ticks = np.linspace(data.t[0], data.t[-1], 4)
    axs[3].set_xticks(x_ticks)

    for ax in axs:
        ax.set_xlim(data.t[0], data.t[-1])

    # Add colorbars
    cbar1 = fig.colorbar(mesh, ax=axs[0], orientation='vertical', fraction=0.08, pad=0.001)
    cbar1.set_label('Median Coherence')

    cbar2 = fig.colorbar(sc, ax=axs[2:], orientation='vertical', fraction=0.08, pad=0.001)
    cbar2.set_label('MdCCM')

    plt.suptitle(f"Array coherence and processing results for {data.source_name}")

    # Save or show the plot
    if save:
        plt.savefig(f"{save_dir}.jpg", dpi=300)

    plt.show()

    return fig, axs