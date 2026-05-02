from .base import Solver
from .common import apply_diffusion_operator
from .recording import observable_matrices, package_recordings
from .runtime import prepare_solver_runtime
from axonscope.axons.base import AxonBase
from axonscope.simresult import SimResult

import jax.numpy as jnp
import jax

# -----------------------------------------------------------------------------
# Euler explicit solver
# -----------------------------------------------------------------------------
class Euler(Solver):
    """
    Explicit forward Euler solver for the cable equation.

    Description
    -----------
    The solver advances voltage using a forward Euler discretization in time.
    The diffusion term uses the standard centered stencil on a uniform mesh and
    its conservative non-uniform counterpart when `x_vec` is irregular, with
    sealed-end (zero-flux) boundary conditions.
    Gating variables are updated using the `update_gates` CNEXP-like helper.

    Discretization (for interior spatial index i):
        d2Vdx2[i] ≈ (V[i+1] - 2 V[i] + V[i-1]) / dx^2
        dV/dt = D * d2Vdx2 + (I_inj - I_ion) / C_m

        V_new = V + dt * dV/dt

    Stability
    ---------
    The explicit Euler scheme is conditionally stable. For diffusion term
    roughly dt <= dx^2 / (2*D) is required. 

    Implementation notes
    --------------------
    - Uses `jax.lax.scan` over time so the kernel can be jitted and it GPU-ready.
    - Carry is the tuple (V, gates); `scan` collects V at each step.
    """

    def __init__(self) -> None:
        """Create an Euler solver instance."""
        pass

    def solve(
        self,
        axon: AxonBase,
        tsim: float,
        dt: float,
        record_observables: bool = False,
    ) -> SimResult:
        """
        Execute an Euler simulation.

        Parameters
        ----------
        axon : AxonBase
            Model containing geometry, ion channel model and stimulation function.
        tsim : float
            Total simulation time in ms.
        dt : float
            Time step in ms.

        Returns
        -------
        SimResult
            object containing (V_all, t_vec) where V_all shape is (Nt, Nx).
        """
        if bool(getattr(axon, "use_extracellular", False)):
            raise NotImplementedError(
                "Euler does not support extracellular coupling. "
                "Use a Crank-Nicholson solver instead."
            )

        runtime = prepare_solver_runtime(
            axon,
            tsim,
            dt,
            include_extracellular=False,
            include_area=False,
        )
        membrane_runtime = runtime.membrane
        grid = runtime.grid
        cable = runtime.cable
        stimulation = runtime.stimulation

        Nt: int = grid.Nt
        backend = membrane_runtime.backend
        membrane = membrane_runtime.membrane
        observable_names = membrane_runtime.observable_names
        dtype_local = membrane_runtime.dtype
        inj_fun = stimulation.intracellular_current_density
        V0: jnp.ndarray = membrane_runtime.Vm0_mV
        gates0: jnp.ndarray = membrane_runtime.gates0
        state0 = membrane_runtime.state0
        t_vec: jnp.ndarray = grid.t_vec_ms

        # Extract ion channel helpers (function objects) once for efficiency
        Cm: float = axon.Cm

        lower, diag, upper = cable.lower, cable.diag, cable.upper
        I_bg = membrane_runtime.background_current

        def step(carry, n: int):
            """
            Single Euler step executed inside `lax.scan`.

            Parameters
            ----------
            carry : tuple (V, gates)
                V : ndarray (Nx,)
                gates : ndarray (Nx, n_gates)
            n : int
                Current time-step index (0..Nt-1)

            Returns
            -------
            carry_out : tuple (V_new, gates_new)
            V_new : ndarray (Nx,)
                Voltage after this time step (also collected by scan).
            """
            V, gates, *state_prev = carry
            state_prev = tuple(state_prev)
            t: float = dtype_local(n) * dt

            # Gating variables update
            gates_new: jnp.ndarray = backend.cn_gate_update(g_prev=gates, V_mV=V, dt=dt)

            # Diffusion uses sealed-end boundary conditions at both cable ends.
            diffusion = apply_diffusion_operator(V, lower, diag, upper)

            # Ionic currents from ion channel model (vectorized)
            Iion = backend.currents(V_mV=V, gates=gates_new)
            step_plan = membrane.prepare_membrane_step(
                V_mV=V,
                gates_prev=gates,
                gates_new=gates_new,
                state=state_prev,
                dt=dt,
                I_ion=Iion,
                I_background=I_bg,
            )

            # External injection current (vectorized over compartments)
            Iinj = inj_fun(t)

            # Time derivative and Euler update
            dVdt = diffusion + (
                Iinj - step_plan.total_outward_current - step_plan.correction_current
            ) / Cm
            V_new = V + dt * dVdt
            state_new = membrane.finalize_membrane_step(
                V_mV_prev=V,
                V_mV_new=V_new,
                gates_prev=gates,
                gates_new=gates_new,
                state_prev=state_prev,
                step_plan=step_plan,
                dt=dt,
            )

            if record_observables:
                gate_obs, current_obs, conductance_obs, state_obs = observable_matrices(
                    membrane, V_new, gates_new, state_new
                )
                return (V_new, gates_new, *state_new), (
                    V_new,
                    gate_obs,
                    current_obs,
                    conductance_obs,
                    state_obs,
                )

            return (V_new, gates_new, *state_new), V_new

        # Run scan: it returns (final_carry, all_outputs)
        _, out = jax.lax.scan(step, (V0, gates0, *state0), jnp.arange(Nt))
        if record_observables:
            V_all = out[0]
            recordings = package_recordings(
                observable_names,
                out[1],
                out[2],
                out[3],
                out[4],
            )
            return SimResult(axon, V_all, t_vec, recordings=recordings)

        V_all = out
        return SimResult(axon, V_all, t_vec)
