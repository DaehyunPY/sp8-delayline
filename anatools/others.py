from typing import Mapping, Callable, Any, Sequence, TypeVar

from cytoolz import curry
from numba import jit
from numpy import array, sin, cos

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


@jit
def rot_mat(th):
    return array(((cos(th), -sin(th)),
                  (sin(th), cos(th))))


@curry
@jit
def affine_transform(x, y, th=0, x0=0, y0=0, dx=1, dy=1, x1=0, y1=0):
    return (rot_mat(th) @ (x, y) - (x0, y0)) * (dx, dy) + (x1, y1)
