from __future__ import annotations

from abc import ABC, abstractmethod

from runtime_common import run_backend_script


class CNRuntimeBenchmark(ABC):
    label: str
    vel_tol: float = 0.01
    err_tol: float = 0.25
    Nx_values = None

    @abstractmethod
    def build_runner(self, problem):
        raise NotImplementedError

    def run(self) -> None:
        run_backend_script(
            label=self.label,
            build_runner=self.build_runner,
            vel_tol=self.vel_tol,
            err_tol=self.err_tol,
            Nx_values=self.Nx_values,
        )
