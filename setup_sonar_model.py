"""
Wrapper function for consistent sonar model setup with eval_u_Sonar_20_const.
"""
import numpy as np
from scipy import sparse as sp
from getParam_Sonar import getParam_Sonar
from eval_u_Sonar import eval_u_Sonar_20_const
from eval_f_Sonar import eval_f_Sonar
from eval_g_Sonar import eval_g_Sonar


def setup_sonar_model_full(
    Nx=301,
    Nz=51,
    Lx=5625,
    Lz=937.5,
    f0=20,
    BC=True,
    UseSparseMatrices=True,
    source_position="surface",
    hydrophone_config="horizontal",
    t_extra=0.0,
):
    """
    Complete sonar model setup with eval_u_Sonar_20_const (constant 20Hz signal).

    Parameters
    ----------
    Nx, Nz : int
        Grid dimensions.
    Lx, Lz : float
        Domain size in meters.
    f0 : float
        Signal frequency in Hz (for wavelength calculations).
    BC : bool
        Use boundary conditions (True recommended).
    UseSparseMatrices : bool
        Use sparse matrices (True recommended for large grids).
    source_position : str or tuple
        'center', 'surface', or explicit (ix, iz).
    hydrophone_config : str or dict
        'horizontal', 'vertical', or dict with x/z indices.
    t_extra : float
        Additional simulation time beyond default (seconds).
    """

    dx = Lx / (Nx - 1)
    dz = Lz / (Nz - 1)
    if dx != dz:
        raise ValueError("ERROR, asymmetric discretization!!!!")

    # Base parameters
    p, x_start, t_start, t_stop, max_dt_FE = getParam_Sonar(
        Nx, Nz, Lx, Lz, UseSparseMatrices=UseSparseMatrices, BC=BC
    )
    t_stop += t_extra

    # Source position
    if source_position == "center":
        p["sonar_ix"] = Nx // 2
        p["sonar_iz"] = Nz // 2
    elif source_position == "surface":
        p["sonar_ix"] = Nx // 2
        p["sonar_iz"] = 1  # just below surface
    elif isinstance(source_position, (tuple, list)) and len(source_position) == 2:
        p["sonar_ix"] = source_position[0]
        p["sonar_iz"] = source_position[1]
    else:
        raise ValueError(f"Unknown source_position: {source_position}")

    # Rebuild B matrix with new source location (pressure assumed in second half per reordered state)
    N = Nx * Nz
    source_idx = p["sonar_ix"] * Nz + p["sonar_iz"]
    B_lil = sp.lil_matrix((2 * N, 1), dtype=float)
    B_lil[source_idx, 0] = 1.0 / (p["dx"] * p["dz"])
    p["B"] = B_lil.tocsr()

    # Hydrophones
    hydro_cfg = hydrophone_config.lower() if isinstance(hydrophone_config, str) else hydrophone_config
    if hydro_cfg == "horizontal":
        z_pos = Nz // 2
        n_phones = 5
        x_indices = [(Nx - 1) * (i + 1) // (n_phones + 1) for i in range(n_phones)]
        p["hydrophones"] = {"z_pos": z_pos, "x_indices": x_indices, "n_phones": n_phones}
    elif hydro_cfg == "vertical":
        x_pos = Nx // 2
        n_phones = 5
        z_indices = [(Nz - 1) * (i + 1) // (n_phones + 1) for i in range(n_phones)]
        p["hydrophones"] = {"x_pos": x_pos, "z_indices": z_indices, "n_phones": n_phones}
    elif isinstance(hydro_cfg, dict):
        p["hydrophones"] = hydro_cfg
    else:
        raise ValueError(f"Unknown hydrophone_config: {hydrophone_config}")

    # Wavelength calculations
    wavelength = p["c"] / f0
    ppw = wavelength / p["dx"]

    # Scaled input
    def eval_u_scaled(t):
        return (p["dx"] * p["dz"]) * eval_u_Sonar_20_const(t)

    model = {
        "p": p,
        "x_start": x_start,
        "t_start": t_start,
        "t_stop": t_stop,
        "max_dt_FE": max_dt_FE,
        "eval_f": eval_f_Sonar,
        "eval_u": eval_u_Sonar_20_const,
        "eval_u_scaled": eval_u_scaled,
        "eval_g": eval_g_Sonar,
        "Nx": Nx,
        "Nz": Nz,
        "Lx": Lx,
        "Lz": Lz,
        "dx": p["dx"],
        "dz": p["dz"],
        "f0": f0,
        "wavelength": wavelength,
        "ppw": ppw,
        "c": p["c"],
        "source_ix": p["sonar_ix"],
        "source_iz": p["sonar_iz"],
        "hydrophones": p["hydrophones"],
    }
    return model


def setup_sonar_model(
    Nx=161,
    Nz=41,
    Lx=4e3,
    Lz=1e3,
    f0=20,
    BC=True,
    UseSparseMatrices=True,
    source_position="surface",
    hydrophone_config="horizontal",
    t_extra=0.0,
):
    """Wrapper for setup_sonar_model_full using smaller grid for quicker runs."""
    return setup_sonar_model_full(
        Nx=Nx,
        Nz=Nz,
        Lx=Lx,
        Lz=Lz,
        f0=f0,
        BC=BC,
        UseSparseMatrices=UseSparseMatrices,
        hydrophone_config=hydrophone_config,
        source_position=source_position,
        t_extra=t_extra,
    )


def print_model_info(model, verbose=True):
    """Print summary of model configuration."""
    print("=" * 70)
    print("SONAR MODEL CONFIGURATION")
    print("=" * 70)

    print(f"\nGrid: {model['Nx']} × {model['Nz']} = {model['Nx']*model['Nz']:,} cells")
    print(f"Domain: {model['Lx']:.0f}m × {model['Lz']:.0f}m")
    print(f"Spacing: dx = {model['dx']:.4f}m, dz = {model['dz']:.4f}m")

    print("\nAcoustic Properties:")
    print(f"  Sound speed: c = {model['c']} m/s")
    print(f"  Frequency: f₀ = {model['f0']} Hz")
    print(f"  Wavelength: λ = {model['wavelength']:.2f}m")
    print(f"  Resolution: {model['ppw']:.1f} points per wavelength")
    print(f"  Domain coverage: {model['Lx']/model['wavelength']:.1f}λ × {model['Lz']/model['wavelength']:.1f}λ")

    source_x = model["source_ix"] * model["dx"]
    source_z = model["source_iz"] * model["dz"]
    print("\nSource Position:")
    print(f"  Grid indices: ({model['source_ix']}, {model['source_iz']})")
    print(f"  Physical: x = {source_x:.1f}m, z = {source_z:.1f}m")

    hydro = model["hydrophones"]
    n_phones = hydro.get("n_phones", len(hydro.get("x_indices", [])))
    print(f"\nHydrophones: {n_phones} receivers")
    if "z_pos" in hydro and "x_indices" in hydro:
        print(f"  Type: Horizontal array at z = {hydro['z_pos'] * model['dz']:.1f}m")
        for i, x_idx in enumerate(hydro["x_indices"]):
            x_pos = x_idx * model["dx"]
            print(f"    H{i+1}: x = {x_pos:.1f}m")
    elif "x_pos" in hydro and "z_indices" in hydro:
        print(f"  Type: Vertical array at x = {hydro['x_pos'] * model['dx']:.1f}m")
        print(f"    Depths: z = {min(hydro['z_indices'])*model['dz']:.1f}m to {max(hydro['z_indices'])*model['dz']:.1f}m")
    elif "x_indices" in hydro and "z_indices" in hydro:
        print("  Type: Custom paired hydrophones (x,z):")
        for i, (xi, zi) in enumerate(zip(hydro["x_indices"], hydro["z_indices"])):
            print(f"    H{i+1}: x = {xi*model['dx']:.1f}m, z = {zi*model['dz']:.1f}m")

    print("\nTime Integration:")
    print(f"  Time span: {model['t_start']:.3f}s to {model['t_stop']:.3f}s ({model['t_stop']*1000:.1f}ms)")
    print(f"  Max stable dt (CFL): {model['max_dt_FE']*1e6:.2f} μs")

    N_states = model["x_start"].shape[0]
    print(f"\nState Vector: {N_states:,} DOFs")

    if verbose:
        print("\nRecommended Settings:")
        nyquist_dt = 1 / (2 * model["f0"])
        if model["max_dt_FE"] <= nyquist_dt:
            print(f"  ✓ CFL timestep satisfies Nyquist (limit = {nyquist_dt*1e6:.2f} μs)")
        else:
            print(f"  ✗ WARNING: CFL > Nyquist! Limit = {nyquist_dt*1e6:.2f} μs")

        if model["ppw"] >= 4:
            print(f"  ✓ Spatial resolution adequate ({model['ppw']:.1f} ppw ≥ 4)")
        else:
            print(f"  ✗ WARNING: Low spatial resolution ({model['ppw']:.1f} ppw < 4)")

    print("=" * 70)


if __name__ == "__main__":
    model = setup_sonar_model()
    print_model_info(model)
