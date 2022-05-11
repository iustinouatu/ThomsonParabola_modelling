import numpy as np
from scipy.constants import c, m_p

def from_KEineV_to_uzinit(KE): # utility function
    """ Converts from Kinetic Energy in eV to the corresponding velocity (relativity taken into account).

    Parameters
    ----------
    KE : float (Kinetic Energy in eV)

    Returns
    -------
    v_total : float

    """

    KE_in_J = from_KEineV_to_KEinJ(KE)
    v_total = ( c / ( (KE_in_J/(m_p*(c**2))) + 1) )  *  np.sqrt( ((KE_in_J/(m_p*(c**2))) + 1)**2 - 1 )
    return v_total


def from_KEineV_to_KEinJ(KE): # utility function
    """ Converts from Kinetic Energy in eV to Kinetic Energy in Joules J.

    Parameters
    ----------
    KE : float (Kinetic Energy in EV)

    Returns
    -------
    KE_in_J : float (Kinetic Energy in J)

    """

    KE_in_J = (1.60217662 * 10**(-19)) * KE # KE in eV
    return KE_in_J