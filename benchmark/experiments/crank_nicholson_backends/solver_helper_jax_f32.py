import jax
import jax.numpy as jnp

dtype = jnp.float32  # tout en float32

# --------------------------
# Safe exponential
# --------------------------
def safe_exp(x):
    x = jnp.asarray(x, dtype=dtype)
    x = jnp.clip(x, a_min=-100.0, a_max=100.0)
    return jnp.exp(x)

# --------------------------
# Current injection
# --------------------------
def Iinj_uAcm2(t, t_start_inj, t_stop_inj, Nx, inj_uA_per_cm2, idx_inj):
    t = jnp.asarray(t, dtype=dtype)
    mask = (t >= t_start_inj) & (t <= t_stop_inj)
    arr = jnp.zeros((Nx,), dtype=dtype)
    arr = jax.lax.cond(
        mask,
        lambda a: a.at[idx_inj].set(dtype(inj_uA_per_cm2)),
        lambda a: a,
        arr
    )
    return arr

# --------------------------
# vtrap
# --------------------------
def vtrap(x, y):
    x = jnp.asarray(x, dtype=dtype)
    y = jnp.asarray(y, dtype=dtype)
    z = x / y
    small = jnp.abs(z) < 1e-6
    exp_term = safe_exp(z)
    denom = exp_term - dtype(1.0)
    safe_val = x / denom
    series = y * (dtype(1.0) - z / dtype(2.0))
    return jnp.where(small, series, safe_val)

# --------------------------
# Ionic currents
# --------------------------
def Iion(V, Nx, m, n, h):
    V = jnp.asarray(V, dtype=dtype)
    if V.shape == ():  # scalar broadcast
        V = jnp.full((Nx,), V, dtype=dtype)

    m = jnp.asarray(m, dtype=dtype)
    n = jnp.asarray(n, dtype=dtype)
    h = jnp.asarray(h, dtype=dtype)

    gnabar = dtype(0.12)
    gkbar = dtype(0.036)
    gl = dtype(0.0003)
    el = dtype(-59.4)
    ena = dtype(45.0)
    ek = dtype(-82.0)

    gna = gnabar * (m ** 3) * h
    gk = gkbar * (n ** 4)

    ina = gna * (V - ena) * dtype(1e3)
    ik = gk * (V - ek) * dtype(1e3)
    il = gl * (V - el) * dtype(1e3)

    return ina + ik + il

# --------------------------
# Gate updates
# --------------------------
def update_gate_halfstep(g_prev, alpha_fun, beta_fun, V, dt):
    V = jnp.asarray(V, dtype=dtype)
    g_prev = jnp.asarray(g_prev, dtype=dtype)
    celsius = dtype(37.0)
    q10 = dtype(2.24659524757) ** ((celsius - dtype(6.3)) / dtype(10.0))
    alpha = q10 * alpha_fun(V)
    beta = q10 * beta_fun(V)
    denom = (dtype(1.0)/dt + dtype(0.5)*(alpha + beta))
    denom = jnp.maximum(denom, dtype(1e-12))
    term1 = alpha / denom
    term2 = ((dtype(1.0)/dt - dtype(0.5)*(alpha + beta)) / denom) * g_prev
    return term1 + term2

def half_step_gates(dt_ms, V_mV, Nx, m, h, n):
    V = jnp.asarray(V_mV, dtype=dtype)
    if V.shape == ():
        V = jnp.full((Nx,), V, dtype=dtype)
    m = jnp.asarray(m, dtype=dtype)
    h = jnp.asarray(h, dtype=dtype)
    n = jnp.asarray(n, dtype=dtype)

    m_new = update_gate_halfstep(m, alpha_m, beta_m, V, dt_ms)
    h_new = update_gate_halfstep(h, alpha_h, beta_h, V, dt_ms)
    n_new = update_gate_halfstep(n, alpha_n, beta_n, V, dt_ms)
    return m_new, h_new, n_new

# --------------------------
# Alpha / Beta functions
# --------------------------
def alpha_m(V): V = jnp.asarray(V, dtype=dtype); return vtrap(dtype(2.5) - dtype(0.1)*(V + dtype(70.0)), dtype(1.0))
def beta_m(V): V = jnp.asarray(V, dtype=dtype); return dtype(4.0) * safe_exp(-(V + dtype(70.0))/dtype(18.0))
def alpha_h(V): V = jnp.asarray(V, dtype=dtype); return dtype(0.07) * safe_exp(-(V + dtype(70.0))/dtype(20.0))
def beta_h(V): V = jnp.asarray(V, dtype=dtype); return dtype(1.0)/(safe_exp(dtype(3.0) - dtype(0.1)*(V + dtype(70.0))) + dtype(1.0))
def alpha_n(V): V = jnp.asarray(V, dtype=dtype); return dtype(0.1) * vtrap(dtype(1.0) - dtype(0.1)*(V + dtype(70.0)), dtype(1.0))
def beta_n(V): V = jnp.asarray(V, dtype=dtype); return dtype(0.125) * safe_exp(-(V + dtype(70.0))/dtype(80.0))

# --------------------------
# Rates
# --------------------------
def rates(V_mV):
    V = jnp.asarray(V_mV, dtype=dtype)
    celsius = dtype(37.0)
    q10 = dtype(2.24659524757) ** ((celsius - dtype(6.3))/dtype(10.0))

    am = alpha_m(V); bm = beta_m(V)
    ah = alpha_h(V); bh = beta_h(V)
    an = alpha_n(V); bn = beta_n(V)

    sum_m = jnp.maximum(am + bm, dtype(1e-12))
    sum_h = jnp.maximum(ah + bh, dtype(1e-12))
    sum_n = jnp.maximum(an + bn, dtype(1e-12))

    mtau = dtype(1.0) / (q10 * sum_m)
    htau = dtype(1.0) / (q10 * sum_h)
    ntau = dtype(1.0) / (q10 * sum_n)

    minf = am / sum_m
    hinf = ah / sum_h
    ninf = an / sum_n

    return minf, mtau, hinf, htau, ninf, ntau
