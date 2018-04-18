from typing import Callable, Optional, NewType, NamedTuple

from numpy import linspace, vectorize, log, pi, sin
from scipy.optimize import curve_fit
from numba import jit

from .others import rot_mat
from .units import as_atomic_mass, as_nano_sec, as_electron_volt


__all__ = ('none_field', 'uniform_electric_field', 'ion_spectrometer', 'electron_spectrometer')


class Hit(NamedTuple):
    """
    Defines a set of detected time, position x and y
    """
    t: float
    x: float
    y: float


class AnalyzedHit(NamedTuple):
    """
    Defines a set of kinetic energy and momentum which are analyzed by a Model
    """
    px: float
    py: float
    pz: float
    ke: float


Model = NewType('Model', Callable[[Hit], Optional[AnalyzedHit]])


class Accelerated(NamedTuple):
    accelerated_momentum: float
    flight_time: float


class Accelerator:
    __accelerate: Callable[..., Optional[Accelerated]]

    def __init__(self, accelerate: Callable[..., Optional[Accelerated]]):
        """
        Initialize an Accelerator. You might want to use this class as a decorator

        :param accelerate: a function(initial_momentum: float) -> Accelerated
        """
        self.__accelerate = accelerate

    def __call__(self, initial_momentum: float, **kwargs) -> Optional[Accelerated]:
        return self.__accelerate(initial_momentum, **kwargs)

    def __mul__(self, other: 'Accelerator') -> Optional['Accelerator']:
        """
        Compose two Accelerator instances. It is order sensitive! Where (self * other)(initial momentum),
        the accelerator 'self' will be called it after 'other' was.

        :param other: an Accelerator
        :return: composed Accelerator
        """

        def accelerate(initial_momentum, **kwargs) -> Optional[Accelerated]:
            acc0 = other(initial_momentum, **kwargs)
            if acc0 is None:
                return None
            acc1 = self(acc0.accelerated_momentum, **kwargs)
            if acc1 is None:
                return None
            return Accelerated(accelerated_momentum=acc1.accelerated_momentum,
                               flight_time=acc0.flight_time + acc1.flight_time)

        return Accelerator(accelerate)


def none_field(length: float) -> Accelerator:
    if length <= 0:
        raise ValueError("Invalid argument 'length'!")

    @Accelerator
    def accelerator(initial_momentum: float, mass: float, **kwargs) -> Optional[Accelerated]:
        if initial_momentum <= 0:
            return None
        if mass <= 0:
            return None
        return Accelerated(accelerated_momentum=initial_momentum, flight_time=length / initial_momentum * mass)

    return accelerator


def uniform_electric_field(length: float, electric_field: float) -> Accelerator:
    if length <= 0:
        raise ValueError("Invalid argument 'length'!")
    if electric_field == 0:
        raise ValueError("Invalid argument 'electric_field'!")

    @Accelerator
    def accelerator(initial_momentum: float, mass: float, charge: float, **kwargs) -> Optional[Accelerated]:
        if mass <= 0:
            return None
        if charge == 0:
            return None
        ke = initial_momentum ** 2 / 2 / mass + electric_field * charge * length
        if ke <= 0:
            return None
        p = (2 * ke * mass) ** 0.5
        t = (p - initial_momentum) / electric_field / charge
        return Accelerated(accelerated_momentum=p, flight_time=t)

    return accelerator


@jit(nopython=True, nogil=True)
def pz_model(t, a5: float, a4: float, a3: float, a2: float, a1: float, a0: float) -> float:
    x = log(t)
    return a5 * x ** 5 + a4 * x ** 4 + a3 * x ** 3 + a2 * x ** 2 + a1 * x ** 1 + a0


def ion_spectrometer(accelerator: Accelerator, mass: float, charge: float, safe_pz_range: float = 400) -> Model:
    p = linspace(-safe_pz_range, safe_pz_range, num=1001)
    _, t = vectorize(accelerator)(p, mass=mass, charge=charge)
    opt, _ = curve_fit(pz_model, t, p)
    diff = p - pz_model(t, *opt)
    print("""------------------------------------------------------
ion model summary
------------------------------------------------------
                    mass (u): {mass_u:10.3f}
                   mass (au): {mass_au:10.3f}
                 charge (au): {charge:6.0f}
    flight time at pz=0 (ns): {flight:10.3f}
time domain of pz model (ns): {tmin:10.3f} -- {tmax:10.3f}
 safe range of pz model (au): {pmin: 6.0f}     -- {pmax:6.0f}
safe max kinetic energy (eV): {kmax:10.3f}
 pz error in the domain (au): {dmin: 10.3f} -- {dmax:10.3f}""".format(
        mass_u=as_atomic_mass(mass),
        mass_au=mass,
        charge=charge,
        flight=as_nano_sec(accelerator(0, mass=mass, charge=charge).flight_time),
        tmin=as_nano_sec(t.min()),
        tmax=as_nano_sec(t.max()),
        pmin=-safe_pz_range,
        pmax=safe_pz_range,
        kmax=as_electron_volt(safe_pz_range ** 2 / 2 / mass),
        dmin=diff.min(),
        dmax=diff.max()))

    @jit(nopython=True, nogil=True)
    def model(hit: Hit) -> AnalyzedHit:
        pz = pz_model(hit.t, opt[0], opt[1], opt[2], opt[3], opt[4], opt[5])
        px = hit.x / hit.t * mass
        py = hit.y / hit.t * mass
        ke = (px ** 2 + py ** 2 + pz ** 2) / 2 / mass
        return AnalyzedHit(px=px, py=py, pz=pz, ke=ke)
    return model


def electron_spectrometer(accelerator: Accelerator, magnetic_filed: float = 0, safe_pz_range: float = 2) -> Model:
    p = linspace(-safe_pz_range, safe_pz_range, num=1001)
    _, t = vectorize(accelerator)(p, mass=1, charge=-1)
    opt, _ = curve_fit(pz_model, t, p)
    diff = p - pz_model(t, *opt)
    print("""------------------------------------------------------
electron model summary
------------------------------------------------------
    flight time at pz=0 (ns): {flight:10.3f}
time domain of pz model (ns): {tmin:10.3f} -- {tmax:10.3f}
 safe range of pz model (au): {pmin: 6.0f}     -- {pmax:6.0f}
safe max kinetic energy (eV): {kmax:10.3f}
 pz error in the domain (au): {dmin: 10.3f} -- {dmax:10.3f}""".format(
        flight=as_nano_sec(accelerator(0, mass=1, charge=-1).flight_time),
        tmin=as_nano_sec(t.min()),
        tmax=as_nano_sec(t.max()),
        pmin=-safe_pz_range,
        pmax=safe_pz_range,
        kmax=as_electron_volt(safe_pz_range ** 2 / 2),
        dmin=diff.min(),
        dmax=diff.max()))

    @jit(nopython=True, nogil=True)
    def model(hit: Hit) -> AnalyzedHit:
        if magnetic_filed == 0:
            th = 0
            pr = 1 / hit.t
        else:
            freq = -magnetic_filed
            th = (freq * hit.t / 2) % pi
            pr = -magnetic_filed / 2 / sin(th)
        pz = pz_model(hit.t, opt[0], opt[1], opt[2], opt[3], opt[4], opt[5])
        px, py = rot_mat(th) @ (hit.x, hit.y) * pr
        ke = (px ** 2 + py ** 2 + pz ** 2) / 2
        return AnalyzedHit(px=px, py=py, pz=pz, ke=ke)
    return model
