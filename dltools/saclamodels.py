from typing import List, Optional

from numba import jit
from numpy import array

from .hittypes import Hit, AnalyzedHit
from .units import to_milli_meter, to_nano_sec, in_atomic_mass

__all__ = ('AModel',)


@jit(nopython=True, nogil=True)
def pr_model(r: float, t: float,
             par0: float, par1: float, par2: float, par3: float, par4: float, par5: float) -> float:
    r = to_milli_meter(r)
    t = to_nano_sec(t)
    return par0 * r + par1 * r * t + par2 * r ** 3 * t + par3 * r ** 5 * t + par4 * t ** 3 + par5 * t ** 5


@jit(nopython=True, nogil=True)
def pz_model(r: float, t: float,
             par0: float, par1: float, par2: float, par3: float, par4: float, par5: float, par6: float) -> float:
    r = to_milli_meter(r)
    t = to_nano_sec(t)
    return par0 + par1 * t + par2 * t ** 2 + par3 * t ** 3 + par4 * t ** 4 + par5 * r ** 2 + par6 * r ** 4


class AModel:
    def __init__(self, mass: float, flight_time_from: float, flight_time_to: float,
                 pr_coeffs: List[float], pz_coeffs: List[float],
                 x_shift: float = 0, y_shift: float = 0):
        self.__fr = flight_time_from
        self.__to = flight_time_to
        self.__x1 = x_shift
        self.__y1 = y_shift
        pr_coeffs = array(pr_coeffs, dtype='float')
        pz_coeffs = array(pz_coeffs, dtype='float')

        # @jit(nopython=True, nogil=True)
        def model(hit: Hit) -> AnalyzedHit:
            px = pr_model(hit.x, hit.t,
                          pr_coeffs[0], pr_coeffs[1], pr_coeffs[2],
                          pr_coeffs[3], pr_coeffs[4], pr_coeffs[5])
            py = pr_model(hit.y, hit.t,
                          pr_coeffs[0], pr_coeffs[1], pr_coeffs[2],
                          pr_coeffs[3], pr_coeffs[4], pr_coeffs[5])
            pz = pz_model((hit.x ** 2 + hit.y ** 2) ** 0.5, hit.t,
                          pz_coeffs[0], pz_coeffs[1], pz_coeffs[2],
                          pz_coeffs[3], pz_coeffs[4], pz_coeffs[5], pz_coeffs[6])
            ke = (px ** 2 + py ** 2 + pz ** 2) / 2 / in_atomic_mass(mass)
            return AnalyzedHit(px=px, py=py, pz=pz, ke=ke)

        self.__model = model

    def __call__(self, t: float, x: float, y: float) -> Optional[dict]:
        """
        :param t: fligt time in nano secs
        :param x: detected x location in milli meters
        :param y: detected y location in milli meters
        """
        if not self.__fr < t < self.__to:
            return None
        return self.__model(Hit.in_experimental_units(t=t, x=x + self.__x1, y=y + self.__y1)).to_experimental_units()
