from typing import Mapping, Sequence, Callable, TypeVar
from operator import getitem, sub
from numpy import nan, pi, sin, cos, array
from scipy.optimize import newton
from cytoolz import curry, flip, reduce, memoize, compose
from .imported import jit
from .tools import call_with_kwargs

T = TypeVar('T')


@curry
@jit
def momentum_xy(x, y, t, magnetic_filed=0, mass=1, charge=0):
    angular_frequency = magnetic_filed * charge / mass
    theta = (angular_frequency * t) % (2 * pi) / 2
    momentum = magnetic_filed * charge / sin(theta) / 2
    return momentum * array(((cos(theta), -sin(theta)),
                             (sin(theta), cos(theta)))) @ (x, y)


@curry
@jit
def accelerate(momentum, electric_filed=0, length=0, mass=1, charge=0) -> Mapping:
    if length <= 0:
        raise ValueError("Parameter 'length' {} is invalid!".format(length))
    if mass <= 0:
        raise ValueError("Parameter 'mass' {} is invalid!".format(mass))

    if charge == 0:
        return {'momentum': nan, 'diff_momentum': nan, 'flight_time': nan, 'diff_flight_time': nan}
    if electric_filed == 0:
        if momentum <= 0:
            return {'momentum': nan, 'diff_momentum': nan, 'flight_time': nan, 'diff_flight_time': nan}
        return {'momentum': momentum,
                'diff_momentum': 1,
                'flight_time': length * mass / momentum,
                'diff_flight_time': - length * mass / momentum**2}

    energy = momentum**2 / 2 / mass + electric_filed * charge * length
    if energy <= 0:
        return {'momentum': nan, 'diff_momentum': nan, 'flight_time': nan, 'diff_flight_time': nan}
    accelerated = (energy*2*mass)**0.5
    return {'momentum': accelerated,
            'diff_momentum': momentum / accelerated,
            'flight_time': (accelerated - momentum) / electric_filed / charge,
            'diff_flight_time': (momentum/accelerated - 1) / electric_filed / charge}


@curry
@jit
def __wrap_accelerator(accelerator: Callable[[T], Mapping],
                       momentum: T, diff_momentum=1, flight_time=0, diff_flight_time=0) -> Mapping:
    accelerated = accelerator(momentum)
    return {'momentum': accelerated['momentum'],
            'diff_momentum': diff_momentum * accelerated['diff_momentum'],
            'flight_time': flight_time + accelerated['flight_time'],
            'diff_flight_time': diff_flight_time + diff_momentum * accelerated['diff_flight_time']}


@curry
@jit
def compose_accelerators(accelerators: Sequence[Callable[[T], Mapping]], momentum: T) -> Mapping:
    return reduce(flip(call_with_kwargs), map(__wrap_accelerator, accelerators), {'momentum': momentum})


@curry
@jit
def momentum_z(accelerator: Callable[[T], Mapping], t: T) -> float:
    memoized = memoize(accelerator)
    return newton(compose(flip(sub, t), flip(getitem, 'flight_time'), memoized), 0,
                  compose(flip(getitem, 'diff_flight_time'), memoized))
