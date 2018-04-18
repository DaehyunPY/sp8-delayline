from typing import Mapping, Callable, Sequence, TypeVar, Any

from cytoolz import curry
from numba import jit
from numpy import array, sin, cos, float64, ndarray

__all__ = ('call', 'call_with_args', 'call_with_kwargs', 'is_between', 'rot_mat', 'affine_transform')

T = TypeVar('T')


@curry
@jit
def call(f: Callable[[T], Any], arg: T):
    return f(arg)


@curry
def call_with_args(f: Callable[[Sequence], Any], args: Sequence):
    return f(*args)


@curry
def call_with_kwargs(f: Callable[[Mapping], Any], kwargs: Mapping):
    return f(**kwargs)


@curry
def is_between(fr, to, v):
    return (fr <= v) & (v <= to)


@jit(nopython=True, nogil=True)
def rot_mat(th: float) -> ndarray:
    return array(((cos(th), -sin(th)),
                  (sin(th), cos(th))), dtype=float64)


@jit(nopython=True, nogil=True)
def affine_transform(x: float, y: float, th: float=0, x0: float=0, y0: float=0, dx: float=1, dy: float=1,
                     x1: float=0, y1: float=0) -> ndarray:
    return ((rot_mat(th) @
             array((x, y), dtype=float64) - array((x0, y0), dtype=float64)) *
            array((dx, dy), dtype=float64) +
            array((x1, y1), dtype=float64))
