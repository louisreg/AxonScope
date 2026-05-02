import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pytest
import os 
from axonscope.axons.unmyelinated import RattayAberham
from axonscope.solvers.crank_nicholson import CrankNicholson
from axonscope.stimulus import Stimulus
from axonscope.simresult import SimResult

pytestmark = pytest.mark.nrv_numerics



# ============================
# 1. CONSTANTS & CONFIGURATION 
# ============================

L = 1000.0                          # Axon length [µm]
Nx = 51                             # Number of compartments (nodes)
d = 1.0                             # Axon diameter [µm]
TSIM = 5.0                         # Total simulation time [ms]
DT = 0.001                          # Time step [ms]
AMPLITUDE = 5                       # Current amplitude [nA]
T_PULSE = 1.0                       # Pulse duration [ms]
T_START = 1.0                       # Pulse start time [ms]
PERTURBATION_FACTOR = 3.0           # Focusing factor for non-uniform mesh
PEAK_TOLERANCE_MV = 0.25            # Off-site peak difference tolerance [mV]
ARRIVAL_TOLERANCE_MS = 0.05         # Off-site peak time difference tolerance [ms]
VELOCITY_RTOL = 0.15                # Average propagation velocity tolerance


def run_ra_simulation(axon: RattayAberham, tsim: float, dt: float) -> SimResult:
    axon.insert_I_Clamp(position=L / 2, stimulus=Stimulus.pulse(start=T_START, duration=T_PULSE, amplitude=AMPLITUDE))
    solver = CrankNicholson()
    res = solver.solve(axon, tsim=tsim, dt=dt)
    return res


def nearest_index(x: np.ndarray, position_um: float) -> int:
    """Return the index of the node closest to a physical position."""
    return int(np.argmin(np.abs(np.asarray(x) - position_um)))


def peak_metrics(res: SimResult, position_um: float) -> tuple[float, float]:
    """
    Return peak amplitude and its occurrence time at a fixed physical position.

    The comparison is intentionally performed away from the injected node because
    the API injects a point current in nA. The equivalent current density scales
    with the local control-volume length, so the local peak at the stimulation
    site is not mesh invariant by construction.
    """
    idx = nearest_index(np.asarray(res.axon.x), position_um)
    trace = np.asarray(res.Vm[:, idx])
    peak_idx = int(np.argmax(trace))
    return float(trace[peak_idx]), float(np.asarray(res.t)[peak_idx])

# ==============================================================================
# 2. MESH CREATION FUNCTION 
# ==============================================================================

def create_focused_non_uniform_x(L: float, Nx: int, perturbation_factor: float) -> jnp.ndarray:
    """
    Creates a non-uniform spatial mesh highly compressed (dense) at the center (L/2).
    Uses the normalized hyperbolic sine function.
    """
    xi = jnp.linspace(-1.0, 1.0, Nx)
    x_transformed = jnp.sinh(perturbation_factor * xi) / jnp.sinh(perturbation_factor)
    x_non_uniform = (x_transformed + 1.0) * L / 2.0
    return x_non_uniform

# ==============================================================================
# 3. PLOTTING FUNCTION (COMBINED WITH GRIDSPEC)
# ==============================================================================

def plot_full_comparison(res_uniform: SimResult, res_non_uniform: SimResult, save_dir: str):
    """
    Generates a single figure summarizing mesh properties and simulation results
    using GridSpec for a non-uniform 3-row layout.
    """
    L_val = res_uniform.axon.x[-1].item()
    
    # --- 1D Comparison Points ---
    x_positions = [L_val/4, L_val/2, 3*L_val/4]
    indices_uniform = [np.argmin(np.abs(res_uniform.axon.x - xp)).item() for xp in x_positions]
    indices_non_uniform = [np.argmin(jnp.abs(res_non_uniform.axon.x - res_uniform.axon.x[idx])).item() 
                           for idx in indices_uniform]
    t_points = [T_START + 0.1, T_START + T_PULSE/2.0, T_START + T_PULSE + 1.0]
    time_indices = [np.argmin(np.abs(res_uniform.t - tp)).item() for tp in t_points]
    
    # --- Mesh Edges Calculation for Pcolormesh ---
    x_centers_uni = res_uniform.axon.x
    dx_uni = x_centers_uni[1] - x_centers_uni[0]
    Y_mesh_uni = np.append(x_centers_uni - dx_uni/2, x_centers_uni[-1] + dx_uni/2)
    
    x_centers_non_uni = res_non_uniform.axon.x
    x_midpoints = (x_centers_non_uni[:-1] + x_centers_non_uni[1:]) / 2
    Y_mesh_non_uni = np.concatenate(([0.0], x_midpoints, [L_val]))
    
    T_mesh = np.append(res_uniform.t, res_uniform.t[-1] + DT)
    
    # --- Setup Figure using GridSpec (The required modification) ---
    fig = plt.figure(figsize=(15, 12)) # Increase width slightly for better fit

    # Define the grid: 3 rows, 2 columns. 
    # Height ratios: Row 1 (mesh) is 1 unit high, Row 2 & 3 (sim) are 4 units high.
    gs = fig.add_gridspec(
        nrows=3, 
        ncols=2, 
        height_ratios=[1, 3, 3], # Makes the top row 4 times shorter than others
        wspace=0.25, 
        hspace=0.35
    )

    # ------------------------------------------------------------------
    # --- ROW 1: MESH COMPARISON (Takes full width) ---
    # ------------------------------------------------------------------
    # Use the first row, spanning both columns (index 0 on the rows, and 0:2 on the columns)
    ax0 = fig.add_subplot(gs[0, 0:2]) 
    
    # 1. Plot Uniform Mesh
    ax0.plot(res_uniform.axon.x, np.zeros_like(res_uniform.axon.x), 
            'o', markersize=4, color='blue', alpha=0.7, label='Uniform grid')
    # 2. Plot Non-Uniform Mesh
    ax0.plot(res_non_uniform.axon.x, np.ones_like(res_non_uniform.axon.x) * 0.5, 
            'x', markersize=5, color='red', alpha=0.9, label='Non uniform grid')
    
    ax0.axvline(L_val / 2, color='gray', linestyle='--', linewidth=1, label='Injection Site (L/2)')
    
    ax0.set_yticks([0, 0.5])
    ax0.set_ylim(-0.1, 0.6)
    ax0.set_yticklabels(['Uniform', 'Non-Uniform'])
    ax0.set_xlim(0, L_val)
    ax0.set_xlabel('Axon x-axis')
    ax0.legend(loc='upper right', fontsize=8)
    ax0.grid(axis='x', linestyle=':')

    # ------------------------------------------------------------------
    # --- ROW 2: 1D SIMULATION COMPARISON ---
    # ------------------------------------------------------------------
    
    ax10 = fig.add_subplot(gs[1, 0]) # Row 2, Col 0: Vm vs Time
    for i in range(len(x_positions)):
        xp = res_uniform.axon.x[indices_uniform[i]].item()
        color = plt.get_cmap("Set1")(i) # Use a colormap for distinct positions
        
        ax10.plot(res_uniform.t, res_uniform.Vm[:, indices_uniform[i]], 
                    label=f'Uniform x={xp:.0f}µm', linestyle='-', alpha=0.7, color=color)
        ax10.plot(res_non_uniform.t, res_non_uniform.Vm[:, indices_non_uniform[i]], 
                    #label=f'Non-Uniform x={res_non_uniform.axon.x[indices_non_uniform[i]]:.0f}µm',
                    linestyle='--', alpha=0.7, color=color)

    ax10.set_ylabel('Vm [mV]')
    ax10.set_xlabel('Time [ms]')
    ax10.legend(fontsize=8, loc='best')
    ax10.grid(True)
    
    ax11 = fig.add_subplot(gs[1, 1]) # Row 2, Col 1: Vm vs Position
    
    for idx_t, tp in zip(time_indices, t_points):
        color = plt.get_cmap("Set1")(idx_t)
        # Uniform mesh line (Solid)
        ax11.plot(res_uniform.axon.x, res_uniform.Vm[idx_t, :], 
                    label=f'Uniform t={tp:.2f}ms', linestyle='-', alpha=0.7, color=color)
        # Focused mesh line (Dashed)
        ax11.plot(res_non_uniform.axon.x, res_non_uniform.Vm[idx_t, :], 
                    linestyle='--', alpha=0.7, color=color) 
    
    ax11.set_ylabel('Vm [mV]')
    ax11.set_xlabel('Position [µm]')
    ax11.legend(fontsize=8, loc='best')
    ax11.grid(True)


    # ------------------------------------------------------------------
    # --- ROW 3: 2D SPACE-TIME MAPS ---
    # ------------------------------------------------------------------
    
    # Use the max/min of both Vm arrays for consistent color scaling
    vmin_global = min(np.min(res_uniform.Vm), np.min(res_non_uniform.Vm))
    vmax_global = max(np.max(res_uniform.Vm), np.max(res_non_uniform.Vm))

    ax20 = fig.add_subplot(gs[2, 0]) # Row 3, Col 0: 2D Uniform
    im_uni = ax20.pcolormesh(
        T_mesh, 
        Y_mesh_uni, 
        res_uniform.Vm.T, 
        cmap='viridis',
        shading='flat',
        vmin=vmin_global, vmax=vmax_global
    )
    ax20.set_title('Uniform Grid')
    ax20.set_ylabel('Position [µm]')
    ax20.set_xlabel('Time [ms]')
    
    ax21 = fig.add_subplot(gs[2, 1]) # Row 3, Col 1: 2D Non-Uniform
    im_non_uni = ax21.pcolormesh(
        T_mesh, 
        Y_mesh_non_uni, 
        res_non_uniform.Vm.T, 
        cmap='viridis',
        shading='flat',
        vmin=vmin_global, vmax=vmax_global
    )
    
    ax21.set_title('Non-Uniform Grid')
    ax21.set_ylabel('Position [µm]')
    ax21.set_xlabel('Time [ms]')

    # Add single colorbar (using the last subplot's mappable)
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.2]) 
    fig.colorbar(im_non_uni, cax=cbar_ax, label='Vm [mV]')

    filename = os.path.join(save_dir, "full_simulation_coherence_plot_gridspec.png")
    fig.savefig(filename)
    plt.close(fig)

# ==============================================================================
# 4. MAIN PYTEST FUNCTION (NO CHANGE NEEDED)
# ==============================================================================

@pytest.mark.parametrize("save_dir", ["figures/physics_tests"])
def test_ra_uniformity(save_dir: str):
    """
    Tests the consistency of the Crank-Nicholson solver by comparing results 
    on a uniform mesh against a highly focused non-uniform mesh, and generates 
    a single 3x2 figure summary.
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # --- 1. Mesh Creation ---
    x_non_uniform = create_focused_non_uniform_x(L, Nx, PERTURBATION_FACTOR)
    
    # --- 2. Run Simulations ---
    axon_uniform = RattayAberham(L=L, d=d, Nx=Nx)
    res_uniform = run_ra_simulation(axon_uniform, tsim=TSIM, dt=DT)
    
    axon_non_uniform = RattayAberham(x_vec=x_non_uniform, d=d, Nx=None)
    res_non_uniform = run_ra_simulation(axon_non_uniform, tsim=TSIM, dt=DT)
    
    # --- 3. Full Comparison Plotting ---
    plot_full_comparison(res_uniform, res_non_uniform, save_dir)
    
    # --- 4. Assertions on mesh-invariant propagation metrics ---
    probe_positions = [L / 4.0, L / 3.0, 2.0 * L / 3.0, 3.0 * L / 4.0]
    peak_diffs = []
    arrival_diffs = []
    for pos in probe_positions:
        peak_uniform, t_uniform = peak_metrics(res_uniform, pos)
        peak_non_uniform, t_non_uniform = peak_metrics(res_non_uniform, pos)
        peak_diffs.append(abs(peak_uniform - peak_non_uniform))
        arrival_diffs.append(abs(t_uniform - t_non_uniform))

    max_peak_diff = max(peak_diffs)
    max_arrival_diff = max(arrival_diffs)
    vel_uniform = float(res_uniform.average_velocity())
    vel_non_uniform = float(res_non_uniform.average_velocity())

    assert max_peak_diff < PEAK_TOLERANCE_MV, (
        f"Peak mismatch away from the stimulus is too large ({max_peak_diff:.4f} mV > "
        f"{PEAK_TOLERANCE_MV} mV)."
    )
    assert max_arrival_diff < ARRIVAL_TOLERANCE_MS, (
        f"Arrival-time mismatch away from the stimulus is too large ({max_arrival_diff:.4f} ms > "
        f"{ARRIVAL_TOLERANCE_MS} ms)."
    )
    assert np.isclose(vel_uniform, vel_non_uniform, rtol=VELOCITY_RTOL), (
        f"Average propagation velocity mismatch is too large ({vel_uniform:.4f} vs "
        f"{vel_non_uniform:.4f} m/s)."
    )

    print(
        f"Test finished. max_peak_diff={max_peak_diff:.4f} mV, "
        f"max_arrival_diff={max_arrival_diff:.4f} ms, "
        f"velocities=({vel_uniform:.4f}, {vel_non_uniform:.4f}) m/s. "
        f"Full plot generated in {save_dir}."
    )
