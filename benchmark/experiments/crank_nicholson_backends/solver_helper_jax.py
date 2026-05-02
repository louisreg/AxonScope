import jax
import jax.numpy as jnp

# --------------------------
# Safe exponential
# --------------------------
def safe_exp(x):
    """
    Safe exponential wrapper that clips large inputs to avoid overflow.
    Uses float64 precision by default when provided inputs are float64.
    """
    x = jnp.asarray(x)
    # clip range chosen to avoid overflow on common hardware
    x = jnp.clip(x, a_min=-100.0, a_max=100.0)
    return jnp.exp(x)


# --------------------------
# Current injection
# --------------------------
def Iinj_uAcm2(t, t_start_inj, t_stop_inj, Nx, inj_uA_per_cm2, idx_inj):
    """
    Return injected current density array [µA/cm²] of length Nx.
    If t is within [t_start_inj, t_stop_inj] then the element at idx_inj
    receives inj_uA_per_cm2, otherwise all zeros.

    This is JAX-friendly (uses .at[...] to set a single index).
    """
    t = jnp.asarray(t)
    mask = (t >= t_start_inj) & (t <= t_stop_inj)
    arr = jnp.zeros((Nx,), dtype=jnp.result_type(inj_uA_per_cm2, jnp.float64))
    # Use scatter update guarded by mask
    arr = jax.lax.cond(
        mask,
        lambda a: a.at[idx_inj].set(inj_uA_per_cm2),
        lambda a: a,
        arr
    )
    return arr


# --------------------------
# vtrap
# --------------------------
def vtrap(x, y):
    """
    Stable vtrap from NEURON modfile.
    returns x / (exp(x/y) - 1) but uses series expansion when z ~ 0.
    Works elementwise for arrays.
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    z = x / y
    small = jnp.abs(z) < 1e-6
    # Avoid calling jnp.exp on huge inputs: clip inside safe_exp
    exp_term = safe_exp(z)
    denom = exp_term - 1.0
    safe_val = x / denom
    series = y * (1.0 - z / 2.0)
    return jnp.where(small, series, safe_val)


# --------------------------
# Ionic currents
# --------------------------
def Iion(V, Nx, m, n, h):
    """
    Return ionic current density [µA/cm²] for V (mV).
    Inputs V, m, n, h can be scalars or 1D arrays (length Nx).
    """
    V = jnp.asarray(V, dtype=jnp.float64)
    # Ensure V has shape (Nx,)
    if V.shape == ():
        V = jnp.full((Nx,), V, dtype=jnp.float64)

    m = jnp.asarray(m, dtype=jnp.float64)
    n = jnp.asarray(n, dtype=jnp.float64)
    h = jnp.asarray(h, dtype=jnp.float64)

    # constants
    gnabar = 0.12   # S/cm^2
    gkbar = 0.036   # S/cm^2
    gl = 0.0003     # S/cm^2
    el = -59.4      # mV
    ena = 45.0      # mV
    ek = -82.0      # mV

    gna = gnabar * (m ** 3) * h
    gk = gkbar * (n ** 4)

    ina = gna * (V - ena) * 1e3
    ik = gk * (V - ek) * 1e3
    il = gl * (V - el) * 1e3

    return ina + ik + il


# --------------------------
# Gate updates
# --------------------------
def update_gate_halfstep(g_prev, alpha_fun, beta_fun, V, dt):
    """
    Half-step gate update used by Crank-Nicholson scheme.
    g_prev, V can be arrays of same shape.
    """
    celsius = 37.0
    q10 = 2.24659524757 ** ((celsius - 6.3) / 10.0)
    alpha = q10 * alpha_fun(V)
    beta = q10 * beta_fun(V)

    # denom might be small, clamp to avoid division by zero
    denom = (1.0 / dt) + 0.5 * (alpha + beta)
    denom = jnp.maximum(denom, 1e-12)

    term1 = alpha / denom
    term2 = ((1.0 / dt) - 0.5 * (alpha + beta)) / denom * g_prev
    return term1 + term2


def half_step_gates(dt_ms, V_mV, Nx, m, h, n):
    """
    Advance gating variables m,h,n one half-step.
    Returns updated (m, h, n) as jnp arrays.
    """
    V = jnp.asarray(V_mV, dtype=jnp.float64)
    # broadcast if scalar
    if V.shape == ():
        V = jnp.full((Nx,), V, dtype=jnp.float64)

    m = jnp.asarray(m, dtype=jnp.float64)
    h = jnp.asarray(h, dtype=jnp.float64)
    n = jnp.asarray(n, dtype=jnp.float64)

    m_new = update_gate_halfstep(m, alpha_m, beta_m, V, dt_ms)
    h_new = update_gate_halfstep(h, alpha_h, beta_h, V, dt_ms)
    n_new = update_gate_halfstep(n, alpha_n, beta_n, V, dt_ms)

    return m_new, h_new, n_new


# --------------------------
# Alpha / Beta functions
# --------------------------
def alpha_m(V_m):
    V_m = jnp.asarray(V_m, dtype=jnp.float64)
    return vtrap(2.5 - 0.1 * (V_m + 70.0), 1.0)


def beta_m(V_m):
    V_m = jnp.asarray(V_m, dtype=jnp.float64)
    return 4.0 * safe_exp(-(V_m + 70.0) / 18.0)


def alpha_h(V_m):
    V_m = jnp.asarray(V_m, dtype=jnp.float64)
    return 0.07 * safe_exp(-(V_m + 70.0) / 20.0)


def beta_h(V_m):
    V_m = jnp.asarray(V_m, dtype=jnp.float64)
    return 1.0 / (safe_exp(3.0 - 0.1 * (V_m + 70.0)) + 1.0)


def alpha_n(V_m):
    V_m = jnp.asarray(V_m, dtype=jnp.float64)
    return 0.1 * vtrap(1.0 - 0.1 * (V_m + 70.0), 1.0)


def beta_n(V_m):
    V_m = jnp.asarray(V_m, dtype=jnp.float64)
    return 0.125 * safe_exp(-(V_m + 70.0) / 80.0)


# --------------------------
# Rates
# --------------------------
def rates(V_mV):
    """
    Compute (minf, mtau, hinf, htau, ninf, ntau) from V.
    V_mV can be scalar or array; returns arrays of same shape.
    """
    v = jnp.asarray(V_mV, dtype=jnp.float64)
    # broadcast scalars to 1D if necessary
    # (if you expect scalar V, you might want scalars back — keep arrays for consistency)
    celsius = 37.0
    q10 = 2.24659524757 ** ((celsius - 6.3) / 10.0)

    am = alpha_m(v)
    bm = beta_m(v)
    ah = alpha_h(v)
    bh = beta_h(v)
    an = alpha_n(v)
    bn = beta_n(v)

    sum_m = jnp.maximum(am + bm, 1e-12)
    sum_h = jnp.maximum(ah + bh, 1e-12)
    sum_n = jnp.maximum(an + bn, 1e-12)

    mtau = 1.0 / (q10 * sum_m)
    htau = 1.0 / (q10 * sum_h)
    ntau = 1.0 / (q10 * sum_n)

    minf = am / sum_m
    hinf = ah / sum_h
    ninf = an / sum_n

    return minf, mtau, hinf, htau, ninf, ntau
