from obspy.geodetics import gps2dist_azimuth
from matplotlib import rcParams
rcParams.update({'font.size': 18,'axes.labelsize': 20, 'axes.titlesize': 14,})
from rtm.plotting import _plot_geographic_context
from rtm.grid import _proj_from_grid
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
from pyproj import CRS, Transformer
import cartopy.crs as ccrs
import colorcet as cc

def plot_network_coherence(Cxy2_net, data, cmin=0.4, cmax=1):
    # --- colormap for coherence ---
    colorm = LinearSegmentedColormap.from_list(
        '', ['white', *plt.get_cmap('magma_r').colors]
    )
    c_lim = [cmin, cmax]

    # --- figure and axis ---
    fig, ax = plt.subplots(figsize=(16, 10))
    plt.subplots_adjust(left=0.06, right=0.94, bottom=0.1, top=0.86)

    # --- main coherence image ---
    mesh = ax.imshow(Cxy2_net, aspect='auto', cmap=colorm, origin='lower',
                     extent=[data.starttime.matplotlib_date,
                             data.endtime.matplotlib_date,
                             data.freq_vector[0],
                             data.freq_vector[-1]],
                     interpolation='none')
    
    mesh.set_clim(c_lim)
    ax.set_ylim(data.freq_min, data.freq_max)
    ax.set_xlim(data.starttime.matplotlib_date, data.endtime.matplotlib_date)
    ax.set_ylabel('Frequency [Hz]')

    # x-axis formatting
    x_ticks = np.linspace(data.starttime.matplotlib_date, data.endtime.matplotlib_date, 5)
    ax.set_xticks(x_ticks)
    ax.xaxis_date()
    if data.n_days <= 1:
        ax.xaxis.set_major_formatter(dates.DateFormatter('%m-%d %H:%M'))
    else:
        ax.xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d %H'))
    ax.tick_params(axis='x')
    ax.grid(alpha=0.8)

    # --- median strip (right side) ---
    median_strip = get_median_strip(Cxy2_net, data, fbin=0.1)
    strip_ax = ax.inset_axes([1.0, 0, 0.03, 1], transform=ax.transAxes)
    median_mesh = strip_ax.imshow(median_strip, aspect='auto', cmap=colorm,
                                  origin='lower', interpolation='none')
    
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
    pair_ax.imshow(mapped[np.newaxis, :], aspect='auto', cmap=greys_cmap, norm=grey_norm,
                   extent=[data.starttime.matplotlib_date,
                           data.endtime.matplotlib_date,
                           0,
                           1],
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

    plt.show()
    return fig, ax



def plot_interstation_coherence(coherograms, st, data, cmin=0.4):
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
    c_lim = [cmin, 1.0]

    fig, axs = plt.subplots(N - 1, N - 1, figsize=(14, 12))
    plt.subplots_adjust(left=0.08, bottom=0.09, hspace=0.1, top=0.95, wspace=0.1)

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

            # now plot
            mesh = ax.imshow(coherogram, aspect='auto', cmap=colorm, origin='lower',
                             extent=[data.t[0],
                                     data.t[-1],
                                     data.freq_vector[0],
                                     data.freq_vector[-1]],
                             interpolation='none')

            mesh.set_clim(c_lim)
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
    cbar.set_label("Mag$^2$ Coherence", fontsize=16)

    # add overarching x-axis label
    fig.text(0.5, 0.02, f'Increasing distance from {data.source_name} --------------->',
             ha='center', va='center', fontsize=16)
    # add overarching y-axis label
    fig.text(0.30, 0.55, f'Increasing distance from {data.source_name} --------------->',
             ha='center', va='center', fontsize=16, rotation=45)

    plt.show()

    return fig, axs


def plot_interstation_phase(phase_pairs, coherence_pairs, phi_obs, coh_weight, st, data):
    station_distances = []
    for tr in st:
        # find distance from source to each station
        dist_m, _, _ = gps2dist_azimuth(data.source_lat, data.source_lon, tr.stats.latitude, tr.stats.longitude)
        station_distances.append((tr.stats.station, dist_m))  # store station name and distance

    stations_sorted_asc = sorted(station_distances, key=lambda x: x[1])  # sort stations by distance ascending

    N = len(stations_sorted_asc)

    colorm = 'twilight_shifted'  # colormap for phase

    fig, axs = plt.subplots(N - 1, N - 1, figsize=(14, 12))
    plt.subplots_adjust(left=0.09, bottom=0.09, hspace=0.1, top=0.95, wspace=0.2, right=0.85)

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
            if (station_y, station_x) in coherence_pairs:
                pair_key = (station_y, station_x)
                coherogram = coherence_pairs[pair_key]
                phase_mat = phase_pairs[pair_key]

            elif (station_x, station_y) in coherence_pairs:
                pair_key = (station_x, station_y)
                coherogram = coherence_pairs[pair_key]
                phase_mat = phase_pairs[pair_key]

            # now plot the main imshow

            coherogram = np.clip(coherogram, 0, 1)
            mesh = ax.imshow(phase_mat, aspect='auto', cmap=colorm, origin='lower',
                             extent=[data.t[0], data.t[-1],
                                     data.freq_vector[0],
                                     data.freq_vector[-1]],
                             interpolation='none', alpha=coherogram)

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

            # Add the phi_obs as a colored bar on the right
            p = data.pairs.index(pair_key)
            phi = phi_obs[p, :]
            coh_alpha = coh_weight[p, :]

            phi_reshaped = phi[:, np.newaxis]  # shape (F, 1)
            coh_reshaped = coh_alpha[:, np.newaxis]

            pos = ax.get_position()
            right_ax = fig.add_axes([pos.x1 + 0.001, pos.y0, 0.01, pos.height])
            right_ax.imshow(phi_reshaped, aspect='auto', cmap=colorm, origin='lower',
                                       extent=[0,
                                               1,
                                               data.freq_min,
                                               data.freq_max],
                                       interpolation='none', vmin=-np.pi,
                                       vmax=np.pi, alpha=coh_reshaped)
            right_ax.set_xlim(0, 1)
            right_ax.set_ylim(data.freq_min, data.freq_max)
            right_ax.set_xticks([])
            right_ax.set_yticks([])


    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(mesh, cax=cbar_ax)  # add colorbar
    cbar.set_label("Observed phase difference [radians]", fontsize=16)
    cbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cbar.set_ticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$'], fontsize=16)

    # add overarching x-axis label
    fig.text(0.5, 0.02, 'Increasing distance from grid center --------------->',
             ha='center', va='center', fontsize=16)
    # add overarching y-axis label
    fig.text(0.30, 0.55, 'Increasing distance from grid center --------------->',
             ha='center', va='center', fontsize=16, rotation=45)

    starttime = st[0].stats.starttime
    endtime = st[0].stats.endtime

    plt.suptitle(f"Inter-station phase"
                 f"\n{starttime.year}-{starttime.month}-{starttime.day} to {endtime.month}-{endtime.day}",
                 fontsize=18)

    plt.show()

    return fig, axs

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


def plot_phase_grid(S, st, dem=None, label_stations=True,
                    cont_int=5, annot_int=50, xy_grid=None,
                    hires=False):
    """
    Plot the 2D phase misfit grid produced by grid_search_phase().
    """
    st = st.copy()
    plt.rcParams.update({'font.size': 13})
    # Get coordinates of grid minimum (best fit)
    min_val = float(S.min().data)
    y_best, x_best = np.unravel_index(np.nanargmin(S.data), S.shape)
    x_best_val = S.x.values[x_best]
    y_best_val = S.y.values[y_best]

    # Get grid center
    lon_0, lat_0 = S.grid_center

    # Handle projections
    if S.UTM:
        projection = None
        transform = None
        proj = _proj_from_grid(S)
        lon_0, lat_0 = proj.transform(S.grid_center[1], S.grid_center[0])

        for tr in st:
            tr.stats.longitude, tr.stats.latitude = proj.transform(
                tr.stats.latitude, tr.stats.longitude
            )

        # Extract corresponding coordinates

        utm_zone = S.attrs['UTM']['zone']
        southern = S.attrs['UTM']['southern_hemisphere']

        # 4️⃣ Construct CRS for the UTM zone
        epsg_code = 32700 + utm_zone if southern else 32600 + utm_zone
        utm_crs = CRS.from_epsg(epsg_code)
        geodetic_crs = CRS.from_epsg(4326)  # WGS84

        # 5️⃣ Build transformer and convert
        transformer = Transformer.from_crs(utm_crs, geodetic_crs, always_xy=True)
        best_lon, best_lat = transformer.transform(x_best_val, y_best_val)
    else:
        projection = ccrs.AlbersEqualArea(
            central_longitude=lon_0,
            central_latitude=lat_0,
            standard_parallels=(S.y.values.min(), S.y.values.max())
        )
        transform = ccrs.PlateCarree()
        best_lon = x_best_val
        best_lat = y_best_val

    # Create figure with two panels
    fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(16, 8),
                                           subplot_kw=dict(projection=projection))

    # Now define plot_transform after axes creation
    if S.UTM:
        plot_transform_main = ax_main.transData
        plot_transform_zoom = ax_zoom.transData
    else:
        plot_transform_main = ccrs.PlateCarree()
        plot_transform_zoom = ccrs.PlateCarree()

    # Compute zoom extent: 1/10th of main grid width, centered on best-fit
    main_width = min((S.x.max() - S.x.min()), (S.y.max() - S.y.min()))
    zoom_half = main_width / 20  # half of 1/10th
    zoom_xmin = x_best_val - zoom_half
    zoom_xmax = x_best_val + zoom_half
    zoom_ymin = y_best_val - zoom_half
    zoom_ymax = y_best_val + zoom_half

    # Slice for zoom
    S_zoom = S.sel(x=slice(zoom_xmin, zoom_xmax), y=slice(zoom_ymin, zoom_ymax))
    if dem is not None:
        dem_zoom = dem.sel(x=slice(zoom_xmin, zoom_xmax), y=slice(zoom_ymin, zoom_ymax))

    # Convert UTM coordinates so center is (0,0) if requested (apply to both main and zoom)
    if xy_grid and S.UTM:
        print(f'Converting to x/y grid, cropping {xy_grid:d} m from center')
        x0 = S.x.data.min() + S.x_radius
        y0 = S.y.data.min() + S.y_radius
        S = S.assign_coords(x=(S.x.data - x0))
        S = S.assign_coords(y=(S.y.data - y0))
        S_zoom = S_zoom.assign_coords(x=(S_zoom.x.data - x0))
        S_zoom = S_zoom.assign_coords(y=(S_zoom.y.data - y0))


        if dem is not None:
            dem = dem.assign_coords(x=(dem.x.data - x0))
            dem = dem.assign_coords(y=(dem.y.data - y0))
            dem_zoom = dem_zoom.assign_coords(x=(dem_zoom.x.data - x0))
            dem_zoom = dem_zoom.assign_coords(y=(dem_zoom.y.data - y0))

        for tr in st:
            tr.stats.longitude -= x0
            tr.stats.latitude -= y0
        x_best_val -= x0
        y_best_val -= y0
        lon_0 -= x0
        lat_0 -= y0
        zoom_xmin -= x0
        zoom_xmax -= x0
        zoom_ymin -= y0
        zoom_ymax -= y0

    # Plot DEM contours if available - main
    if dem is not None:
        # Rounding to nearest cont_int
        all_levels = np.arange(np.ceil(dem.min().data / cont_int),
                               np.floor(dem.max().data / cont_int) + 1) * cont_int
        # Rounding to nearest annot_int
        annot_levels = np.arange(np.ceil(dem.min().data / annot_int),
                                 np.floor(dem.max().data / annot_int) + 1) * annot_int
        # Ensure we don't draw annotated levels twice
        cont_levels = [lvl for lvl in all_levels if lvl not in annot_levels]

        dem.plot.contour(ax=ax_main, colors='k', levels=cont_levels, linewidths=0.3, zorder=-1)
        cs = dem.plot.contour(ax=ax_main, colors='k', levels=annot_levels, linewidths=0.7, zorder=-1)
        ax_main.clabel(cs, fontsize=9, fmt='%d', inline=True)

        alpha = 0.55  # change this

        # Mask areas outside of DEM extent
        dem_slice = dem.sel(x=S.x, y=S.y, method='nearest')
        S.data[np.isnan(dem_slice.data)] = np.nan
    else:
        if not S.UTM:
            _plot_geographic_context(ax=ax_main, hires=hires)
            alpha = 0.5
        else:
            alpha = 1

    # Plot phase misfit grid - main
    cmap = cc.cm.fire
    vmin = float(S.min().data)
    vmax = float(S.min().data + (S.max().data - S.min().data) * 0.5)  # using max as 3/4 for better contrast
    alpha_data = 0.9 * (1.0 - (S.data - vmin) / (vmax - vmin + 1e-12))
    alpha_data = np.clip(alpha_data, 0.05, 0.9)  # never fully transparent or >0.85

    alpha_data_zoom = 0.9 * (1.0 - (S_zoom.data - vmin) / (vmax - vmin + 1e-12))
    alpha_data_zoom = np.clip(alpha_data_zoom, 0.05, 0.9)

    if S.UTM:
        sm_main = S.plot.imshow(ax=ax_main, cmap=cmap, alpha=alpha_data, add_colorbar=False, vmin=vmin, vmax=vmax)
        if xy_grid:
            ax_main.set_xlabel('X [m]')
            ax_main.set_ylabel('Y [m]')
        else:
            ax_main.set_xlabel('UTM easting [m]')
            ax_main.set_ylabel('UTM northing [m]')
            ax_main.ticklabel_format(style='plain', useOffset=False)
    else:
        sm_main = S.plot.pcolormesh(ax=ax_main, transform=transform, cmap=cmap, alpha=alpha, add_colorbar=False,
                                    vmin=vmin, vmax=vmax)

    # Plot DEM contours if available - zoom
    if dem is not None:
        all_levels_zoom = np.arange(np.ceil(dem.min().data / (cont_int / 2)),
                                    np.floor(dem.max().data / (cont_int / 2)) + 1) * (cont_int / 2)
        # Rounding to nearest annot_int
        annot_levels_zoom = np.arange(np.ceil(dem.min().data / (annot_int / 2)),
                                      np.floor(dem.max().data / (annot_int / 2)) + 1) * (annot_int / 2)
        # Ensure we don't draw annotated levels twice
        cont_levels_zoom = [lvl for lvl in all_levels_zoom if lvl not in annot_levels_zoom]

        dem_zoom.plot.contour(ax=ax_zoom, colors='k', levels=cont_levels_zoom, linewidths=0.3, zorder=-1)
        cs_zoom = dem_zoom.plot.contour(ax=ax_zoom, colors='k', levels=annot_levels_zoom, linewidths=0.7, zorder=-1)
        ax_zoom.clabel(cs_zoom, fontsize=9, fmt='%d', inline=True)

        # Mask for zoom
        dem_slice_zoom = dem_zoom.sel(x=S_zoom.x, y=S_zoom.y, method='nearest')
        S_zoom.data[np.isnan(dem_slice_zoom.data)] = np.nan
    else:
        if not S.UTM:
            _plot_geographic_context(ax=ax_zoom, hires=hires)

    # Plot phase misfit grid - zoom
    if S.UTM:
        S_zoom.plot.imshow(ax=ax_zoom, cmap=cmap, alpha=alpha_data_zoom, add_colorbar=False, vmin=vmin,
                                     vmax=vmax)
        if xy_grid:
            ax_zoom.set_xlabel('X [m]')
            ax_zoom.set_ylabel('Y [m]')
        else:
            ax_zoom.set_xlabel('UTM easting [m]')
            ax_zoom.set_ylabel('UTM northing [m]')
            ax_zoom.ticklabel_format(style='plain', useOffset=False)
    else:
        S_zoom.plot.pcolormesh(ax=ax_zoom, transform=transform, cmap=cmap, alpha=alpha, add_colorbar=False,
                                         vmin=vmin, vmax=vmax)

    # Set zoom extent (for cartopy, after plotting)
    if not S.UTM:
        ax_zoom.set_extent([zoom_xmin, zoom_xmax, zoom_ymin, zoom_ymax], crs=transform)
    else:
        ax_zoom.set_xlim(zoom_xmin, zoom_xmax)
        ax_zoom.set_ylim(zoom_ymin, zoom_ymax)

    # Add black square on main plot showing zoom extent
    from matplotlib.patches import Rectangle
    zoom_width = zoom_xmax - zoom_xmin
    zoom_height = zoom_ymax - zoom_ymin
    rect = Rectangle((zoom_xmin, zoom_ymin), zoom_width, zoom_height, edgecolor='black',
                     facecolor='none', linewidth=1.2, transform=plot_transform_main)
    ax_main.add_patch(rect)

    # Initialize list of handles for legend (only on main)
    scatter_zorder = 5
    h = [None, None, None]

    # Plot grid center - both axes
    ax_main.scatter(lon_0, lat_0, s=50, color='limegreen', edgecolor='black',
                    label='Grid center', transform=plot_transform_main, zorder=scatter_zorder)
    ax_zoom.scatter(lon_0, lat_0, s=50, color='limegreen', edgecolor='black',
                    label='Grid center', transform=plot_transform_zoom, zorder=scatter_zorder)
    h[0] = ax_main.collections[-1]  # Get handle from main

    # Plot best-fit source (minimum misfit) - both
    if S.UTM:
        label = 'Best fit'
    else:
        label = f'Best fit\n({y_best_val:.4f}, {x_best_val:.4f})'
    ax_main.scatter(x_best_val, y_best_val, s=150, color='white', marker='*',
                    edgecolor='black', label=label, transform=plot_transform_main, zorder=scatter_zorder)
    ax_zoom.scatter(x_best_val, y_best_val, s=150, color='white', marker='*',
                    edgecolor='black', label=label, transform=plot_transform_zoom, zorder=scatter_zorder)
    h[1] = ax_main.collections[-1]

    # Plot stations - both
    for tr in st:
        ax_main.scatter(tr.stats.longitude, tr.stats.latitude, marker='v', color='orange',
                        edgecolor='black', label='Station', transform=plot_transform_main, zorder=scatter_zorder)
        ax_zoom.scatter(tr.stats.longitude, tr.stats.latitude, marker='v', color='orange',
                        edgecolor='black', label='Station', transform=plot_transform_zoom, zorder=scatter_zorder)
        if label_stations:
            ax_main.text(tr.stats.longitude, tr.stats.latitude,
                         f'  {tr.stats.network}.{tr.stats.station}',
                         fontsize=10, color='white', transform=plot_transform_main,
                         verticalalignment='center_baseline', horizontalalignment='left',
                         zorder=scatter_zorder,
                         path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()],
                         clip_on=True)
            ax_zoom.text(tr.stats.longitude, tr.stats.latitude,
                         f'  {tr.stats.network}.{tr.stats.station}',
                         fontsize=10, color='white', transform=plot_transform_zoom,
                         verticalalignment='center_baseline', horizontalalignment='left',
                         zorder=scatter_zorder,
                         path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()],
                         clip_on=True)
    h[2] = ax_main.collections[-1]  # Approximate handle

    # Legend only on main
    ax_main.legend(h, [handle.get_label() for handle in h if handle is not None],
                   loc='lower left', framealpha=1, borderpad=0.3, handletextpad=0.3)

    # Label and style
    title = f'Phase misfit grid\nMin misfit = {min_val:.3f} rad'
    if hasattr(S, 'celerity'):
        title += f'\nCelerity: {S.celerity:g} m/s'

    if S.UTM:
        ax_main.set_aspect('equal')
        ax_zoom.set_aspect('equal')
    else:
        ax_main.set_xlabel('Longitude')
        ax_main.set_ylabel('Latitude')
        ax_zoom.set_xlabel('Longitude')
        ax_zoom.set_ylabel('Latitude')

    ax_main.text(0.02, 0.95, "a)", transform=ax_main.transAxes, fontweight="bold",
                 fontsize=14, bbox=dict(facecolor="white", alpha=0.7))
    ax_zoom.text(0.02, 0.95, "b)", transform=ax_zoom.transAxes, fontweight="bold",
                 fontsize=14, bbox=dict(facecolor="white", alpha=0.7))
    
    ax_zoom.text(0.97, 0.97, f"Best fit\nLat: {best_lat:.5f}\nLon: {best_lon:.5f}",
                 transform=ax_zoom.transAxes, ha="right", va="top", fontsize=16,
                 bbox=dict(facecolor="lightgrey", edgecolor="black", alpha=0.8))
    
    ax_main.grid(alpha=0.5)
    ax_zoom.grid(alpha=0.5)

    # Add shared colorbar on the right
    fig.subplots_adjust(right=0.86, left=0.065, top=0.98, bottom=0.05)
    pos = ax_zoom.get_position()

    cax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height])
    cbar = fig.colorbar(sm_main, cax=cax, label='Mean phase misfit [radians]')
    cbar.solids.set_alpha(1)

    if xy_grid:
        ax_main.set_xlim(-xy_grid, xy_grid)
        ax_main.set_ylim(-xy_grid, xy_grid)

    plt.show()

    return fig