from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np
from scipy.signal import find_peaks

from axonscope.axons import Axon

@dataclass
class SimResult():
    axon: Axon
    Vm: NDArray
    t: NDArray
    
    def rasterize(self, threshold: float = -10.0, min_distance: float = 1.0) -> list[list[float]]:
        """
        Detect action potentials (spikes) and return their timestamps per section.

        Parameters
        ----------
        threshold : float, optional
            Minimum height of Vm peak to be considered an AP (in mV).
            Default = -10.0 mV.
        min_distance : float, optional
            Minimum refractory distance between two APs (in ms).
            Default = 1.0 ms.

        Returns
        -------
        spikes : list of list of float
            spikes[i] is a list of spike times (ms) detected in section i.
        """
        _, Nx = self.Vm.shape
        spikes: list[list[float]] = []

        dt = float(self.t[1] - self.t[0])
        min_distance_pts = int(min_distance / dt)

        for j in range(Nx):
            # detect peaks
            peaks, _ = find_peaks(
                self.Vm[:, j],
                height=threshold,
                distance=min_distance_pts,
            )
            # convert peak indices -> times
            spike_times = self.t[peaks].tolist()
            spikes.append(spike_times)

        return spikes

    def rasterplot(self, ax, threshold: float = -10.0, min_distance: float = 1.0) -> None:
        """
        Plot a raster diagram: section index (y) vs spike times (x).
        """
        spikes = self.rasterize(threshold=threshold, min_distance=min_distance)
        for j, train in enumerate(spikes):
            xpos = self.axon.x[j]  # spatial position of this compartment
            ax.vlines(train, xpos - 0.5, xpos + 0.5, color="black", linewidth=1)

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Axon position (µm)")


    def average_velocity(self, threshold: float = -10.0, min_distance: float = 1.0) -> float:
            """
            Estimate action potential propagation velocity from the raster.

            Returns
            -------
            float
                Propagation velocity in units of distance / time (e.g., mm/ms).
            """
            TODO
            return(0.0)
