import jax.numpy as jnp
import numpy as np
import pytest
from axonscope.axons.unmyelinated import RattayAberham 
from axonscope.solvers.crank_nicholson import CrankNicholson
from axonscope.simresult import SimResult
from axonscope.stimulus import Stimulus

# ==============================================================================
# 1. CONSTANTS & CONFIGURATION
# ==============================================================================

L = 1000.0                          # Axon length [µm]
Nx = 101                            # Number of compartments (Nodes)
d = 1.0                             # Axon diameter [µm]
TSIM = 5.0                          # Total simulation time [ms]
DT = 0.001                          # Time step [ms]
AMPLITUDE = 5.0                     # Current amplitude [nA]
T_PULSE = 1.0                       # Pulse duration [ms]
T_START = 1.0                       # Pulse start time [ms]
PERTURBATION_FACTOR = 1.0           # Focusing factor for the non-uniform mesh
TOLERANCE_MAX_DIFF_COHERENCE = 0.2  # Off-site peak tolerance [mV]
TOLERANCE_ARRIVAL_DIFF_COHERENCE = 0.01  # Off-site peak-time tolerance [ms]
TOLERANCE_VELOCITY_RTOL = 0.01      # Relative tolerance on propagation speed
TOLERANCE_MAX_DIFF_INIT = 1e-8      # Strict tolerance for initialization (L,Nx vs x_vec) [mV]

# ==============================================================================
# 2. NON-UNIFORM MESH FUNCTION
# ==============================================================================

def create_focused_non_uniform_x(L: float, Nx: int, perturbation_factor: float) -> jnp.ndarray:
    """
    Creates a non-uniform spatial mesh strongly compressed (focused) at the center (L/2).
    
    This uses a hyperbolic sine function to smoothly vary the grid spacing.
    """
    xi = jnp.linspace(-1.0, 1.0, Nx)
    # Apply the sinh transformation
    x_transformed = jnp.sinh(perturbation_factor * xi) / jnp.sinh(perturbation_factor)
    # Rescale to the physical length [0, L]
    x_non_uniform = (x_transformed + 1.0) * L / 2.0
    return x_non_uniform.astype(jnp.float32)

# ==============================================================================
# 3. SIMULATION EXECUTION FUNCTION
# ==============================================================================

def run_ra_simulation(axon: RattayAberham, tsim: float, dt: float) -> SimResult:
    """
    Sets up the stimulus and executes the simulation using the CrankNicholson solver.
    """
    # Inject a current clamp in the middle of the axon
    axon.insert_I_Clamp(
        position=L / 2,
        stimulus=Stimulus.pulse(start=T_START, duration=T_PULSE, amplitude=AMPLITUDE),
    )
    solver = CrankNicholson() 
    res = solver.solve(axon, tsim=tsim, dt=dt)
    return res


def nearest_index(x: np.ndarray, position_um: float) -> int:
    """Return the node index closest to a physical position."""
    return int(np.argmin(np.abs(np.asarray(x) - position_um)))


def peak_metrics(res: SimResult, position_um: float) -> tuple[float, float]:
    """
    Return peak amplitude and occurrence time at a fixed physical position.

    We avoid the stimulated node itself because the injected current is specified
    as a point current in nA and is converted to a density using the local
    control-volume length. The local stimulation-site peak therefore depends on
    the mesh by construction.
    """
    idx = nearest_index(np.asarray(res.axon.x), position_um)
    trace = np.asarray(res.Vm[:, idx])
    peak_idx = int(np.argmax(trace))
    return float(trace[peak_idx]), float(np.asarray(res.t)[peak_idx])

# ==============================================================================
# 4. UNIT TEST: UNIFORMITY AND COHERENCE
# ==============================================================================
def test_mesh_coherence():
    """
    Verifies two core consistency scenarios:
    1. Initialization Coherence: RattayAberham(L,Nx) vs RattayAberham(x_vec uniform)
    2. Physical Coherence: Uniform Mesh vs Non-Uniform Mesh results
    """
    
    # ------------------------------------------------------------------
    # MESH SETUP
    # ------------------------------------------------------------------
    # Uniform mesh (reference grid)
    x_uniform = jnp.linspace(0.0, L, Nx, dtype=jnp.float32)

    # Non-uniform (focused) mesh
    x_non_uniform = create_focused_non_uniform_x(L, Nx, PERTURBATION_FACTOR)
    
    # ------------------------------------------------------------------
    # SCENARIO 1: UNIFORM INITIALIZATION TEST (STRICT)
    # ------------------------------------------------------------------
    
    # Axon A: Initialization via L and Nx (internal uniform grid generation)
    axon_L_Nx = RattayAberham(L=L, d=d, Nx=Nx)
    
    # Axon B: Initialization via x_vec (external uniform grid provided)
    axon_x_vec = RattayAberham(x_vec=x_uniform, d=d, Nx=None)
    
    print("\n--- Running Uniform Initialization Test (L,Nx vs x_vec) ---")
    
    # Run both uniform simulations
    res_L_Nx = run_ra_simulation(axon_L_Nx, tsim=TSIM, dt=DT)
    res_x_vec = run_ra_simulation(axon_x_vec, tsim=TSIM, dt=DT)
    
    # Check the maximum absolute difference across all time steps and compartments
    V_diff_init = np.abs(res_L_Nx.Vm - res_x_vec.Vm)
    max_diff_init = np.max(V_diff_init)
    
    print(f"Max Difference (L,Nx vs x_vec): {max_diff_init:.10e} mV")

    # Assert that the difference is near machine precision
    assert max_diff_init < TOLERANCE_MAX_DIFF_INIT, (
        f"🚨 Initialization FAILED. Max difference ({max_diff_init:.10e} mV) "
        f"exceeds strict tolerance ({TOLERANCE_MAX_DIFF_INIT:.10e} mV). "
        "RattayAberham(L,Nx) is not mathematically equivalent to RattayAberham(x_vec uniform)."
    )
    print("✅ Uniform Initialization PASSED.")


    # ------------------------------------------------------------------
    # SCENARIO 2: PHYSICAL COHERENCE TEST (UNIFORM vs NON-UNIFORM)
    # ------------------------------------------------------------------

    # Axon C: Non-uniform mesh (focused grid)
    axon_non_uniform = RattayAberham(x_vec=x_non_uniform, d=d, Nx=None) 
    
    print("\n--- Running Focused Mesh Simulation (Coherence Test) ---")
    
    # Run the non-uniform simulation
    res_non_uniform = run_ra_simulation(axon_non_uniform, tsim=TSIM, dt=DT)
    
    probe_positions = [L / 4.0, L / 3.0, 2.0 * L / 3.0, 3.0 * L / 4.0]
    peak_differences = []
    arrival_differences = []
    for pos in probe_positions:
        peak_uniform, t_uniform = peak_metrics(res_L_Nx, pos)
        peak_non_uniform, t_non_uniform = peak_metrics(res_non_uniform, pos)
        peak_differences.append(abs(peak_uniform - peak_non_uniform))
        arrival_differences.append(abs(t_uniform - t_non_uniform))

    max_difference_coherence = max(peak_differences)
    max_arrival_difference = max(arrival_differences)
    vel_uniform = float(res_L_Nx.average_velocity())
    vel_non_uniform = float(res_non_uniform.average_velocity())

    # Assertion of Physical Consistency
    print("\n--- Coherence Check (Uniform vs Non-Uniform) ---")
    print(f"Max off-site peak difference: {max_difference_coherence:.4f} mV")
    print(f"Max off-site arrival difference: {max_arrival_difference:.4f} ms")
    print(f"Velocity Uniform (Ref): {vel_uniform:.4f} m/s")
    print(f"Velocity Focused: {vel_non_uniform:.4f} m/s")
    
    assert max_difference_coherence < TOLERANCE_MAX_DIFF_COHERENCE, (
        f"🚨 Coherence FAILED. The off-site peak difference ({max_difference_coherence:.4f} mV) "
        f"exceeds the set tolerance ({TOLERANCE_MAX_DIFF_COHERENCE} mV)."
    )
    assert max_arrival_difference < TOLERANCE_ARRIVAL_DIFF_COHERENCE, (
        f"🚨 Coherence FAILED. The off-site arrival difference ({max_arrival_difference:.4f} ms) "
        f"exceeds the set tolerance ({TOLERANCE_ARRIVAL_DIFF_COHERENCE} ms)."
    )
    assert np.isclose(vel_uniform, vel_non_uniform, rtol=TOLERANCE_VELOCITY_RTOL), (
        f"🚨 Coherence FAILED. Velocity mismatch ({vel_uniform:.4f} vs {vel_non_uniform:.4f} m/s) "
        f"exceeds rtol={TOLERANCE_VELOCITY_RTOL}."
    )

    print("Physical Coherence PASSED. Max difference is within tolerance.")
    print("\n--- All tests completed successfully. ---")
