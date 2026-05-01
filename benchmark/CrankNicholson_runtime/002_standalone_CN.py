import numpy as np

from benchmark_base import CNRuntimeBenchmark


def _vtrap(x, y):
    z = x / y
    return np.where(np.abs(z) < 1e-6, y * (1.0 - z / 2.0), x / (np.exp(z) - 1.0))


def _hh_rates(V):
    alpha = np.stack(
        [
            0.1 * _vtrap(-(V + 40.0), 10.0),
            0.07 * np.exp(-(V + 65.0) / 20.0),
            0.01 * _vtrap(-(V + 55.0), 10.0),
        ],
        axis=-1,
    )
    beta = np.stack(
        [
            4.0 * np.exp(-(V + 65.0) / 18.0),
            1.0 / (np.exp(-(V + 35.0) / 10.0) + 1.0),
            0.125 * np.exp(-(V + 65.0) / 80.0),
        ],
        axis=-1,
    )
    return alpha, beta


def _update_gates(problem, gates, V):
    alpha, beta = _hh_rates(V)
    alpha = problem.q10 * alpha
    beta = problem.q10 * beta
    denom = np.maximum(1.0 / problem.dt + 0.5 * (alpha + beta), 1e-12)
    return alpha / denom + ((1.0 / problem.dt) - 0.5 * (alpha + beta)) / denom * gates


def _currents(problem, V, gates):
    g_open = np.stack(
        [
            problem.g_bar[0] * gates[:, 0] ** 3 * gates[:, 1],
            problem.g_bar[1] * gates[:, 2] ** 4,
            np.full(problem.Nx, problem.g_bar[2], dtype=np.float64),
            np.full(problem.Nx, problem.g_bar[3], dtype=np.float64),
        ],
        axis=-1,
    )
    return np.sum(g_open * (V[:, None] - problem.e_rev[None, :]), axis=1)


def _apply_diffusion(problem, V):
    dx_cm = float(np.asarray(problem.axon.h_cm)[0])
    coeff = float(problem.axon.D) / (dx_cm ** 2)
    LV = np.empty_like(V)
    LV[0] = 2.0 * coeff * (V[1] - V[0])
    LV[1:-1] = coeff * (V[:-2] - 2.0 * V[1:-1] + V[2:])
    LV[-1] = 2.0 * coeff * (V[-2] - V[-1])
    return LV


class StandaloneCNBenchmark(CNRuntimeBenchmark):
    label = "standalone"
    Nx_values = [11, 21, 51]

    def build_runner(self, problem):
        Nx = problem.Nx
        Nt = problem.Nt
        dt = problem.dt
        Cm = float(problem.axon.Cm)
        V0 = np.full(Nx, float(problem.axon.Vinit), dtype=np.float64)
        alpha0, beta0 = _hh_rates(V0)
        gates0 = alpha0 / np.maximum(alpha0 + beta0, 1e-12)

        dx_cm = float(np.asarray(problem.axon.h_cm)[0])
        alpha = 0.5 * dt * float(problem.axon.D) / (dx_cm ** 2)
        A = np.zeros((Nx, Nx), dtype=np.float64)
        np.fill_diagonal(A, 1.0 + 2.0 * alpha)
        for i in range(1, Nx - 1):
            A[i, i - 1] = -alpha
            A[i, i + 1] = -alpha
        A[0, 1] = -2.0 * alpha
        A[-1, -2] = -2.0 * alpha

        def run():
            V = V0.copy()
            gates = gates0.copy()
            Vm = np.zeros((Nt, Nx), dtype=np.float64)
            t = (np.arange(Nt, dtype=np.float64) + 1.0) * dt
            for n in range(Nt):
                t_mid = n * dt + 0.5 * dt
                gates = _update_gates(problem, gates, V)
                Iion = _currents(problem, V, gates)
                Iinj = np.zeros(Nx, dtype=np.float64)
                if problem.t_start_inj <= t_mid <= problem.t_stop_inj:
                    Iinj[problem.idx_inj] = problem.inj_uA_per_cm2
                rhs = V + 0.5 * dt * _apply_diffusion(problem, V) + (dt / Cm) * (Iinj - Iion)
                V = np.linalg.solve(A, rhs)
                Vm[n] = V
            return t, Vm

        return run


if __name__ == "__main__":
    StandaloneCNBenchmark().run()
