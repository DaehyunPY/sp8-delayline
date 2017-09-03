from typing import Mapping, Callable, Any, Sequence, TypeVar
from cytoolz import curry
from .imported import jit

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
