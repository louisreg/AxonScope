import numpy as np

def PSA(x_positions, x0, z, I_stim, sigma):
    """
    Compute the extracellular potential along an axon due to a point-source electrode.
    
    Parameters
    ----------
    z_positions : array_like
        Positions along the axon (m).
    x0 : float
        Electrode position along the axon (m).
    I_stim : float
        Stimulation current (A).
    sigma : float
        Conductivity of the medium (S/m).
    d_perp : float
        Perpendicular distance from axon to electrode (m).

    Returns
    -------
    V_ext : ndarray
        Extracellular potential at each x position (V).
    """
    r = np.sqrt((x_positions - x0)**2 + z**2)
    V_ext = I_stim / (4 * np.pi * sigma * r)
    return V_ext


