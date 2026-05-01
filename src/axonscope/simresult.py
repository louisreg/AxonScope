from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from axonscope.axons.base import AxonBase


RecordingDict = Dict[str, Dict[str, NDArray]]

@dataclass
class SimResult():
    axon: AxonBase
    Vm: NDArray
    t: NDArray
    diagnostics: Optional[Dict[str, Any]] = None
    recordings: Optional[RecordingDict] = None
    metadata: Optional[Dict[str, Any]] = None

    def spatial_positions_um(self) -> np.ndarray:
        """Return the spatial positions represented by the Vm columns."""
        metadata = self.metadata or {}
        if "x_um" in metadata:
            return np.asarray(metadata["x_um"], dtype=float)
        return np.asarray(self.axon.x, dtype=float)
    
    def rasterize(
        self, threshold: float = -10.0, min_distance: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect action potentials and return their times and positions.

        Returns
        -------
        tAP : np.ndarray
            Array of spike times (ms)
        xAP : np.ndarray
            Array of spatial positions corresponding to each spike
        """
        x_um = self.spatial_positions_um()
        Nx = self.Vm.shape[1]
        if self.t.shape[0] < 2:
            return np.array([]), np.array([])
        dt = float(self.t[1] - self.t[0])
        min_distance_pts = int(min_distance / dt)

        tAP = []
        xAP = []

        for j in range(Nx):
            # detect peaks in this compartment
            peaks, _ = find_peaks(
                self.Vm[:, j],
                height=threshold,
                distance=min_distance_pts,
            )
            # append peak times and positions
            tAP.extend(self.t[peaks])
            xAP.extend([x_um[j]] * len(peaks))

        return np.array(tAP), np.array(xAP)

    def rasterplot(self, ax, threshold: float = -10.0, min_distance: float = 1.0) -> None:
        """
        Plot a raster diagram: axon position (y) vs spike times (x).
        """
        tAP, xAP = self.rasterize(threshold=threshold, min_distance=min_distance)

        if len(tAP) == 0:
            return  # rien à tracer

        # chaque spike est une ligne verticale très fine centrée sur sa position spatiale
        ax.vlines(tAP, xAP - 0.5, xAP + 0.5, color="black", linewidth=1)

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Axon position (µm)")



    def average_velocity(self, threshold: float = -10.0, min_distance: float = 1.0) -> float:
        """
        Estimate the average velocity in m/s of the action potential along the axon
        using linear regression between origin spike and axon ends.
        """
        tAP, xAP = self.rasterize(threshold=threshold, min_distance=min_distance)

        if len(tAP) == 0:
            return 0.0

        t_flat = np.array(tAP)/1e3      #in s
        x_flat = np.array(xAP)/1e6      #in m

        # Sort spikes by time
        sort_idx = np.argsort(t_flat)
        t_flat = t_flat[sort_idx]
        x_flat = x_flat[sort_idx]

        # First detected spike as origin
        x0 = x_flat[0]

        x_um = self.spatial_positions_um()
        x_min, x_max = x_um[0], x_um[-1]

        # Forward velocity (toward x_max)
        mask_forward = (x_flat >= x0) & (x_flat <= x_max)
        v_forward = 0.0
        if np.sum(mask_forward) >= 2:
            t_sel = t_flat[mask_forward]
            x_sel = x_flat[mask_forward]
            sort_idx = np.argsort(t_sel)
            t_sel = t_sel[sort_idx]
            x_sel = x_sel[sort_idx]
            coeff_forward = np.polyfit(t_sel, x_sel, 1)
            #print(x_sel)
            #print(t_sel)
            #print(np.mean(x_sel/t_sel))
            #exit()
            v_forward = coeff_forward[0]

        # Backward velocity (toward x_min)
        mask_backward = (x_flat <= x0) & (x_flat >= x_min)
        v_backward = 0.0
        if np.sum(mask_backward) >= 2:
            t_sel = t_flat[mask_backward]
            x_sel = x_flat[mask_backward]
            sort_idx = np.argsort(t_sel)
            t_sel = t_sel[sort_idx]
            x_sel = x_sel[sort_idx]
            x_sel = x_sel[::-1]
            coeff_backward = np.polyfit(t_sel, x_sel, 1)
            v_backward = coeff_backward[0]

        # Average of forward/backward
        velocities = [v for v in [v_forward, v_backward] if v != 0.0]
        if len(velocities) == 0:
            return 0.0
        return np.mean(velocities)
