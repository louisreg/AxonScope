import numpy as np

def safe_exp(x):
    #x = np.clip(x, -1000, 1000)  # avoid overflow
    return np.exp(x)

def Iinj_uAcm2(t, t_start_inj, t_stop_inj, Nx, inj_uA_per_cm2, idx_inj):
    if t_start_inj <= t <= t_stop_inj:
        arr = np.zeros(Nx)
        arr[idx_inj] = inj_uA_per_cm2
        return arr
    else:
        return np.zeros(Nx)

def vtrap(x, y):
    """Stable implementation of vtrap (from NEURON mod file)."""
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        z = x / y
        out = np.where(np.abs(z) < 1e-6,
                       y * (1.0 - z / 2.0),   # series expansion
                       x / (np.exp(z) - 1.0))
    return out

# ---- ionic currents ----
def Iion(V, Nx, m, n, h):
    """Return ionic current density [µA/cm²] at time t (ms) for V (mV)."""
    V_arr = np.asarray(V, dtype=float)

    gnabar=0.12   # S/cm^2
    gkbar=0.036   # S/cm^2
    gl=0.0003     # S/cm^2
    el=-59.4      # mV
    ena=45.0      # mV
    ek=-82.0      # mV
    
    gna = gnabar * (m ** 3) * h   # S/cm^2
    gk = gkbar * (n ** 4)              # S/cm^2
    gl = gl                                 # S/cm^2

    ina = gna * (V_arr - ena) * 1e3  # µA/cm^2
    ik = gk * (V_arr - ek) * 1e3
    il = gl * (V_arr - el) * 1e3

    return ina + ik + il

def update_gate_halfstep(g_prev, alpha_fun, beta_fun, V, dt):
        celsius = 37
        q10 = 2.24659524757**((celsius - 6.3)/10)
        alpha = q10 * alpha_fun(V)
        beta  = q10 * beta_fun(V)
        denom = (1.0/dt) + 0.5*(alpha + beta)
        term1 = alpha / denom
        term2 = ((1.0/dt) - 0.5*(alpha + beta)) / denom * g_prev
        return term1 + term2

def half_step_gates(dt_ms, V_mV, Nx, m, h, n):
    if dt_ms <= 0.0:
        return
    
    V = np.asarray(V_mV, dtype=float)
    if V.shape != (Nx,):
        V = np.full(Nx, float(V.item()))

    m = update_gate_halfstep(m, alpha_m, beta_m, V, dt_ms)
    h = update_gate_halfstep(h, alpha_h, beta_h, V, dt_ms)
    n = update_gate_halfstep(n, alpha_n, beta_n, V, dt_ms)
    return(m, h, n)

def alpha_m(V_m):
    return(vtrap(2.5 - 0.1 * (V_m + 70.0), 1.0))

def beta_m(V_m):
    return(4.0 * safe_exp(-(V_m + 70.0) / 18.0))

def alpha_h(V_m):
    return(0.07 * safe_exp(-(V_m + 70.0) / 20.0))

def beta_h(V_m):
    return(1.0 / (safe_exp(3.0 - 0.1 * (V_m + 70.0)) + 1.0))    

def alpha_n(V_m):
    return(0.1 * vtrap(1.0 - 0.1 * (V_m + 70.0), 1.0))

def beta_n(V_m):
    return(0.125 * safe_exp(-(V_m+70)/80))  

def rates(V_mV):
    v = np.asarray(V_mV, dtype=float)

    celsius = 37
    q10 = 2.24659524757**((celsius - 6.3)/10)
    
    # m
    sum_m = np.maximum(alpha_m(v) + beta_m(v), 1e-12)
    mtau = 1.0 / (q10 * sum_m)
    minf = alpha_m(v) / sum_m

    # h
    sum_h = np.maximum(alpha_h(v) + beta_h(v), 1e-12)
    htau = 1.0 / (q10 * sum_h)
    hinf = alpha_h(v) / sum_h

    # n
    sum_n = np.maximum(alpha_n(v) + beta_n(v), 1e-12)
    ntau = 1.0 / (q10 * sum_n)
    ninf = alpha_n(v) / sum_n

    return minf, mtau, hinf, htau, ninf, ntau

