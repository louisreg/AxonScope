import torch

# --------------------------
# Safe exponential
# --------------------------
def safe_exp(x):
    return torch.exp(x)

# --------------------------
# Current injection
# --------------------------
def Iinj_uAcm2(t, t_start_inj, t_stop_inj, Nx, inj_uA_per_cm2, idx_inj):
    arr = torch.zeros(Nx, dtype=torch.float64, device=inj_uA_per_cm2.device if isinstance(inj_uA_per_cm2, torch.Tensor) else 'cpu')
    if t_start_inj <= t <= t_stop_inj:
        arr[idx_inj] = inj_uA_per_cm2
    return arr

# --------------------------
# vtrap
# --------------------------
def vtrap(x, y):
    z = x / y
    out = torch.where(torch.abs(z) < 1e-6,
                      y * (1.0 - z / 2.0),
                      x / (torch.exp(z) - 1.0))
    return out

# --------------------------
# Ionic currents
# --------------------------
def Iion(V, Nx, m, n, h):
    V_arr = V if isinstance(V, torch.Tensor) else torch.tensor(V, dtype=torch.float64)

    gnabar = 0.12
    gkbar  = 0.036
    gl_val = 0.0003
    el     = -59.4
    ena    = 45.0
    ek     = -82.0

    gna = gnabar * m**3 * h
    gk  = gkbar * n**4
    gl  = gl_val

    ina = gna * (V_arr - ena) * 1e3
    ik  = gk * (V_arr - ek) * 1e3
    il  = gl * (V_arr - el) * 1e3

    return ina + ik + il

# --------------------------
# Gate updates
# --------------------------
def update_gate_halfstep(g_prev, alpha_fun, beta_fun, V, dt):
    celsius = 37.0
    q10 = 2.24659524757**((celsius - 6.3)/10)
    alpha = q10 * alpha_fun(V)
    beta  = q10 * beta_fun(V)
    denom = (1.0/dt) + 0.5*(alpha + beta)
    term1 = alpha / denom
    term2 = ((1.0/dt) - 0.5*(alpha + beta)) / denom * g_prev
    return term1 + term2

def half_step_gates(dt_ms, V_mV, Nx, m, h, n):
    if dt_ms <= 0.0:
        return m, h, n

    V = V_mV if isinstance(V_mV, torch.Tensor) else torch.tensor(V_mV, dtype=torch.float64)
    if V.shape[0] != Nx:
        V = V.expand(Nx)

    m = update_gate_halfstep(m, alpha_m, beta_m, V, dt_ms)
    h = update_gate_halfstep(h, alpha_h, beta_h, V, dt_ms)
    n = update_gate_halfstep(n, alpha_n, beta_n, V, dt_ms)

    return m, h, n

# --------------------------
# Alpha / Beta functions
# --------------------------
def alpha_m(V_m):
    return vtrap(2.5 - 0.1 * (V_m + 70.0), 1.0)

def beta_m(V_m):
    return 4.0 * safe_exp(-(V_m + 70.0) / 18.0)

def alpha_h(V_m):
    return 0.07 * safe_exp(-(V_m + 70.0) / 20.0)

def beta_h(V_m):
    return 1.0 / (safe_exp(3.0 - 0.1 * (V_m + 70.0)) + 1.0)

def alpha_n(V_m):
    return 0.1 * vtrap(1.0 - 0.1 * (V_m + 70.0), 1.0)

def beta_n(V_m):
    return 0.125 * safe_exp(-(V_m + 70.0)/80)

# --------------------------
# Rates
# --------------------------
def rates(V_mV):
    v = V_mV if isinstance(V_mV, torch.Tensor) else torch.tensor(V_mV, dtype=torch.float64)

    celsius = 37.0
    q10 = 2.24659524757**((celsius - 6.3)/10)

    # m
    sum_m = torch.clamp(alpha_m(v) + beta_m(v), min=1e-12)
    mtau = 1.0 / (q10 * sum_m)
    minf = alpha_m(v) / sum_m

    # h
    sum_h = torch.clamp(alpha_h(v) + beta_h(v), min=1e-12)
    htau = 1.0 / (q10 * sum_h)
    hinf = alpha_h(v) / sum_h

    # n
    sum_n = torch.clamp(alpha_n(v) + beta_n(v), min=1e-12)
    ntau = 1.0 / (q10 * sum_n)
    ninf = alpha_n(v) / sum_n

    return minf, mtau, hinf, htau, ninf, ntau
