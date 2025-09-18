import matplotlib as mpl
import matplotlib.pyplot as plt
import colorcet as cc
mpl.use('Qt5Agg')

def plot_misfit_latlon(misfit_grid, lon_vec, lat_vec, idx_best, station_coords, save_dir, save=False):
    fig, ax = plt.subplots(figsize=(12, 10))
    pcm = ax.pcolormesh(lon_vec, lat_vec, misfit_grid.T, shading='auto', cmap=cc.cm.fire_r)
    for code, (lon, lat) in station_coords.items():
        ax.plot(lon, lat, '^', markersize=12, markerfacecolor='whitesmoke', markeredgecolor='k')
        ax.text(lon + 0.012 * (lon_vec.max() - lon_vec.min()), lat - 0.01 * (lat_vec.max() - lat_vec.min()),
                code, fontsize=10, color='whitesmoke')

    ax.plot(lon_vec[idx_best[0]], lat_vec[idx_best[1]], '*', color='hotpink', markeredgecolor='k', markersize=15,
            label='Best-fit source', alpha=0.85)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("")

    fig.colorbar(pcm, ax=ax, label='Phase misfit')
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    ax.legend()
    if save:
        fig.savefig(f"{save_dir}.png", dpi=300)
    plt.show()
    return fig, ax