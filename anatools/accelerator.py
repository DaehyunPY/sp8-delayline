from operator import getitem, sub
from typing import Mapping, Sequence, Callable, TypeVar, Iterable

from cytoolz import curry, reduce, compose, memoize
from cytoolz.curried import flip
from numba import jit
from numpy import linspace
from numpy import nan, pi, sin
from scipy.optimize import newton

from .others import call_with_kwargs, rot_mat

__all__ = ('accelerate', 'compose_accelerators', 'momentum')

T = TypeVar('T')
PINFO = Mapping
Accelerator = Callable[[T, T, T], PINFO]


@curry
@jit
def momentum_xy(mass, charge, x, y, t, magnetic_filed=0):
    if magnetic_filed == 0:
        th = 0
        p = mass / t
    else:
        freq = magnetic_filed * charge / mass
        th = (freq * t / 2) % pi
        p = magnetic_filed * charge / sin(th) / 2
    return rot_mat(th) @ (x, y) * p


@curry
@jit
def accelerate(mass, charge, pz, electric_filed=0, length=0) -> PINFO:
    if length <= 0:
        raise ValueError("Parameter 'length' {} is invalid!".format(length))
    if mass <= 0:
        raise ValueError("Parameter 'mass' {} is invalid!".format(mass))

    if charge == 0:
        return {'momentum': nan, 'diff_momentum': nan, 'flight_time': nan, 'diff_flight_time': nan}
    if electric_filed == 0:
        if pz <= 0:
            return {'momentum': nan, 'diff_momentum': nan, 'flight_time': nan, 'diff_flight_time': nan}
        return {'momentum': pz,
                'diff_momentum': 1,
                'flight_time': length * mass / pz,
                'diff_flight_time': - length * mass / pz ** 2}

    energy = pz ** 2 / 2 / mass + electric_filed * charge * length
    if energy <= 0:
        return {'momentum': nan, 'diff_momentum': nan, 'flight_time': nan, 'diff_flight_time': nan}
    accelerated = (energy * 2 * mass) ** 0.5
    return {'momentum': accelerated,
            'diff_momentum': pz / accelerated,
            'flight_time': (accelerated - pz) / electric_filed / charge,
            'diff_flight_time': (pz / accelerated - 1) / electric_filed / charge}


@curry
@jit
def wrap_accelerator(accelerator: Accelerator, mass: T, charge: T,
                     momentum: T, diff_momentum=1, flight_time=0, diff_flight_time=0) -> PINFO:
    accelerated = accelerator(mass, charge, momentum)
    return {'momentum': accelerated['momentum'],
            'diff_momentum': diff_momentum * accelerated['diff_momentum'],
            'flight_time': flight_time + accelerated['flight_time'],
            'diff_flight_time': diff_flight_time + diff_momentum * accelerated['diff_flight_time']}


def compose_accelerators(accelerators: Iterable[Accelerator]) -> Callable[[T, T, T], PINFO]:
    wrapped: Sequence[Callable[[T, T], Callable[[PINFO], PINFO]]] = tuple(wrap_accelerator(acc) for acc in accelerators)

    @curry
    def accelerator(mass, charge, pz) -> PINFO:
        return reduce(flip(call_with_kwargs), (w(mass, charge) for w in wrapped), {'momentum': pz})

    return accelerator


@curry
@jit
def momentum_z(mass, charge, t, accelerator: Accelerator):
    memoized = memoize(accelerator(mass, charge))
    return newton(compose(flip(sub, t), flip(getitem, 'flight_time'), memoized), 0,
                  compose(flip(getitem, 'diff_flight_time'), memoized))


@curry
@jit
def kinetic_energy(px, py, pz, mass=1):
    return (px ** 2 + py ** 2 + pz ** 2) / 2 / mass


@curry
@jit
def momentum(x, y, t, accelerator, magnetic_filed=0, mass=1, charge=0):
    px, py = momentum_xy(mass, charge, x, y, t, magnetic_filed=magnetic_filed)
    pz = momentum_z(mass, charge, t, accelerator=accelerator)
    ke = kinetic_energy(px, py, pz, mass=mass)
    return ke, px, py, pz
