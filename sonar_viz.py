import numpy as np
import matplotlib.pyplot as plt


def plot_pressure_xz_at(p, X, t, index=None, time_s=None, cmap='RdBu_r', sym=True, overlay=True):
    """
    Plot the pressure field p(x,z) across the XZ plane at a given time.

    Parameters
    - p: dict of simulation parameters (expects keys 'Nx','Nz','Lx','Lz','dx','dz', and optional 'hydrophones')
    - X: ndarray of shape (2*N, T), state over time where first N entries are pressure
    - t: 1D array-like of length T with time stamps (seconds)
    - index: optional integer frame index to plot
    - time_s: optional float time (s); nearest index is selected if provided
    - cmap: matplotlib colormap name
    - sym: if True, use symmetric color limits around zero
    - overlay: if True, overlay source and hydrophone positions

    Returns (fig, ax)
    """
    Nx, Nz = int(p['Nx']), int(p['Nz'])
    Lx, Lz = float(p['Lx']), float(p['Lz'])
    dx, dz = float(p['dx']), float(p['dz'])

    N = Nx * Nz
    t_arr = np.asarray(t).reshape(-1)

    # Determine index
    if index is None:
        if time_s is None:
            index = len(t_arr) - 1
        else:
            index = int(np.argmin(np.abs(t_arr - float(time_s))))
    index = int(max(0, min(index, X.shape[1] - 1)))

    # Extract and reshape pressure field
    x_i = np.asarray(X[:, index]).reshape(-1)
    field = x_i[:N].reshape(Nx, Nz).T  # plot as (Z, X)

    vmin = vmax = None
    if sym:
        m = float(np.nanmax(np.abs(field))) if field.size else 1.0
        vmin, vmax = -m, m

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    im = ax.imshow(
        field,
        extent=[0, Lx, Lz, 0],  # Z increases downward
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect='equal',
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Pressure (Pa)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title(f'Pressure field on XZ plane at t={t_arr[index]*1000:.3f} ms (idx {index})')

    if overlay:
        try:
            # Plot source
            sx = p['sonar_ix'] * dx
            sz = p['sonar_iz'] * dz
            ax.plot([sx], [sz], marker='*', color='yellow', markersize=12, markeredgecolor='k')
            
            # Plot hydrophones (handle both horizontal and vertical arrays)
            hp = p.get('hydrophones', {})
            if hp.get('n_phones', 0) > 0:
                if 'z_pos' in hp and 'x_indices' in hp:
                    # Horizontal array: same z, varying x
                    zpos = hp['z_pos'] * dz
                    for x_idx in hp['x_indices']:
                        ax.plot(x_idx * dx, zpos, '^', color='white', markersize=6, markeredgecolor='k', alpha=0.85)
                elif 'x_pos' in hp and 'z_indices' in hp:
                    # Vertical array: same x, varying z
                    xpos = hp['x_pos'] * dx
                    for z_idx in hp['z_indices']:
                        ax.plot(xpos, z_idx * dz, '^', color='white', markersize=6, markeredgecolor='k', alpha=0.85)
        except Exception:
            pass

    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()
    return fig, ax

