from operator import rshift
from textwrap import dedent
from typing import Mapping, Callable, Tuple

from cytoolz import reduce
from numba import jit
from numpy import pi, sin, linspace, ndarray
from scipy.optimize import curve_fit
from sympy import symbols, sqrt, Integer, lambdify

from .others import rot_mat
from .units import as_atomic_mass, as_nano_sec

__all__ = ('accelerator', 'Momentum')

_mass, _charge, _init_momentum, _field, _length = symbols('m q p f l', real=True)

_fin_momentum_wo_filed = _init_momentum
_diff_fin_momentum_wo_filed = Integer(1)
_flight_time_wo_filed = _length * _mass / _init_momentum
_diff_flight_time_wo_filed = - _length * _mass / _init_momentum ** 2

_fin_ke = _init_momentum ** 2 / _mass / 2 + _field * _charge * _length
_fin_momentum = sqrt(2 * _fin_ke * _mass)
_diff_fin_momentum = _init_momentum / _fin_momentum
_flight_time = (_fin_momentum - _init_momentum) / _field / _charge
_diff_flight_time = (_init_momentum / _fin_momentum - 1) / _field / _charge


class Accelerator:
    def __init__(self, fin_momentum, diff_fin_momentum, flight_time, diff_flight_time):
        self.__fin_momentum = fin_momentum
        self.__diff_fin_momentum = diff_fin_momentum
        self.__flight_time = flight_time
        self.__diff_flight_time = diff_flight_time

    def __repr__(self):
        return dedent("""\
            accelerated momentum: {}
            momentum differential: {}
            flight time: {}
            flight time differential: {}""".format(repr(self.fin_momentum),
                                                   repr(self.diff_fin_momentum),
                                                   repr(self.flight_time),
                                                   repr(self.diff_flight_time)))

    @property
    def fin_momentum(self):
        return self.__fin_momentum

    @property
    def diff_fin_momentum(self):
        return self.__diff_fin_momentum

    @property
    def flight_time(self):
        return self.__flight_time

    @property
    def diff_flight_time(self):
        return self.__diff_flight_time

    def __rshift__(self, shift: 'Accelerator') -> 'Accelerator':
        fin_momentum = shift.fin_momentum.subs(_init_momentum, self.fin_momentum)
        diff_fin_momentum = shift.diff_fin_momentum.subs(_init_momentum, self.fin_momentum)
        flight_time = shift.flight_time.subs(_init_momentum, self.fin_momentum)
        diff_flight_time = shift.diff_flight_time.subs(_init_momentum, self.fin_momentum)
        return Accelerator(fin_momentum=fin_momentum,
                           diff_fin_momentum=self.diff_fin_momentum * diff_fin_momentum,
                           flight_time=self.flight_time + flight_time,
                           diff_flight_time=self.diff_flight_time + self.diff_fin_momentum * diff_flight_time)

    def __lshift__(self, shift: 'Accelerator') -> 'Accelerator':
        return shift >> self

    def __mul__(self, other: 'Accelerator') -> 'Accelerator':
        return self << other

    def __call__(self, mass: float, charge: float) -> Callable[[float], float]:  # init momentum -> flight time
        if mass <= 0:
            raise ValueError("Parameter 'mass' {} is invalid!".format(mass))
        if charge == 0:
            raise ValueError("Parameter 'charge' {} is invalid!".format(charge))
        flight_time = self.flight_time.subs(_mass, mass).subs(_charge, charge)
        # diff_flight_time = self.diff_flight_time.subs(_mass, mass).subs(_charge, charge)
        return lambdify(_init_momentum, flight_time, 'numpy')


def single_accelerator(electric_filed: float, length: float) -> Accelerator:
    if length <= 0:
        raise ValueError("Parameter 'length' {} is invalid!".format(length))
    if electric_filed == 0:
        return Accelerator(fin_momentum=_fin_momentum_wo_filed.subs(_length, length),
                           diff_fin_momentum=_diff_fin_momentum_wo_filed.subs(_length, length),
                           flight_time=_flight_time_wo_filed.subs(_length, length),
                           diff_flight_time=_diff_flight_time_wo_filed.subs(_length, length))
    return Accelerator(fin_momentum=_fin_momentum.subs(_field, electric_filed).subs(_length, length),
                       diff_fin_momentum=_diff_fin_momentum.subs(_field, electric_filed).subs(_length, length),
                       flight_time=_flight_time.subs(_field, electric_filed).subs(_length, length),
                       diff_flight_time=_diff_flight_time.subs(_field, electric_filed).subs(_length, length))


def accelerator(*spec: Mapping[str, float]) -> Accelerator:
    if len(spec) == 0:
        raise ValueError('These is no argument!')
    if len(spec) == 1:
        return single_accelerator(**spec[0])
    return reduce(rshift, (single_accelerator(**reg) for reg in spec))


@jit
def momentum_xy(x: float, y: float, t: float, mass: float = 1, charge: float = -1, magnetic_filed: float = 0):
    if magnetic_filed == 0:
        th = 0
        p = mass / t
    else:
        freq = magnetic_filed * charge / mass
        th = (freq * t / 2) % pi
        p = magnetic_filed * charge / sin(th) / 2
    return rot_mat(th) @ (x, y) * p


@jit
def kinetic_energy(px, py, pz, mass: float = 1):
    return (px ** 2 + py ** 2 + pz ** 2) / 2 / mass


class Momentum:
    def __init__(self, accelerator: Accelerator, magnetic_filed: float, mass: float, charge: float):
        self.__mass = mass
        self.__charge = charge
        self.__magnetic_filed = magnetic_filed
        if mass > 1:  # ion
            safe = abs(mass*charge)**0.5*2.5//100*100
            p = linspace(-safe, safe, num=1001)
            acc = accelerator(mass=mass, charge=charge)
            t: ndarray = acc(p)
            self.__model_args, _ = curve_fit(self.model, t, p)
            diff = p - self.model(t, *self.model_args)
            print(dedent("""\
                         momentum calculator summary:
                             mass (u): {m:1.3f}
                             charge (au): {q:1.0f}
                             flight time at pz=0 (ns): {t:1.3f}
                             time domain of pz model (ns): {tmin:1.3f} -- {tmax:1.3f}
                             safe region of pz model (au): -{safe:1.0f} -- {safe:1.0f}
                             pz error in the domain (au): {pmin:1.3f} -- {pmax:1.3f}""".format(
                m=as_atomic_mass(mass),
                q=charge,
                t=as_nano_sec(acc(0)),
                tmin=as_nano_sec(t.min()),
                tmax=as_nano_sec(t.max()),
                safe=safe,
                pmin=diff.min(),
                pmax=diff.max())))
        else:  # electron
            p = linspace(-5, 5, num=1001)
            acc = accelerator(mass=mass, charge=charge)
            t: ndarray = acc(p)
            self.__model_args, _ = curve_fit(self.model, t, p)
            diff = p - self.model(t, *self.model_args)
            print(dedent("""\
                         momentum calculator summary:
                             mass (au): {m:1.3f}
                             charge (au): {q:1.0f}
                             flight time at pz=0 (ns): {t:1.3f}
                             time domain of pz model (ns): {tmin:1.3f} -- {tmax:1.3f}
                             safe region of pz model (au): -5 -- 5
                             pz error in the domain (au): {pmin:1.3f} -- {pmax:1.3f}""".format(
                m=mass,
                q=charge,
                t=as_nano_sec(acc(0)),
                tmin=as_nano_sec(t.min()),
                tmax=as_nano_sec(t.max()),
                pmin=diff.min(),
                pmax=diff.max())))

    def __repr__(self):
        return dedent("""\
            accelerator: unsaved
            magnetic_filed: {}
            mass: {}
            charge: {}""".format(self.mass, self.charge, self.magnetic_filed))

    @property
    def mass(self):
        return self.__mass

    @property
    def charge(self):
        return self.__charge

    @property
    def magnetic_filed(self):
        return self.__magnetic_filed

    @staticmethod
    @jit
    def model(t, a7: float, a6: float, a5: float, a4: float, a3: float, a2: float, a1: float, a0: float):
        return (a7 * t ** 7 + a6 * t ** 6 + a5 * t ** 5 +
                a4 * t ** 4 + a3 * t ** 3 + a2 * t ** 2 + a1 * t + a0)

    @property
    def model_args(self):
        return self.__model_args

    def __call__(self, x, y, t) -> Tuple[float, float, float, float]:  # ke, px, py, pz
        px, py = momentum_xy(x, y, t, mass=self.mass, charge=self.charge, magnetic_filed=self.magnetic_filed)
        pz = self.model(t, *self.model_args)
        ke = kinetic_energy(px, py, pz, mass=self.mass)
        return ke, px, py, pz
