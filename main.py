from yaml import load
from itertools import repeat
from typing import TypeVar, Sequence, Mapping, Callable, Iterator
from cytoolz import concat, concatv, pipe, curry, compose
from cytoolz.curried import take, map, merge, merge_with, filter
from numba import jit
from pathos.multiprocessing import Pool
from analysis.reader import affine_transform, Read
from analysis.tools import call_with_kwargs, call_with_args, is_between, call
from analysis.accelerator import accelerate, compose_accelerators, momentum_xy, momentum_z

T = TypeVar('T')

with open('config.yaml', 'r') as f:
    config = load(f)


@curry
def read(treename: str, filename: str) -> Iterator[Mapping]:
    with Read(filename) as f:
        yield from f[treename]


@curry
def event_filter(filters: Mapping, event: Mapping) -> bool:
    for k, f in {k: is_between(fr, to) for k, (fr, to) in filters.items()}.items():
        if k not in event:
            return False
        if not f(event[k]):
            return False
    return True
global_filter_with_keys = {k: event_filter(config[k].get('global_filter', {})) for k in ('ion', 'electron')}


@curry
@jit
def transformer(transform_t: Callable[[T], T], transform_xy: Callable[[T, T], Sequence[T]], event: Mapping) -> Mapping:
    t = transform_t(event['t'])
    x, y = transform_xy(event['x'], event['y'])
    return merge(event, {'x': x, 'y': y, 't': t})


def transformers(config: Mapping) -> Sequence[Callable[[Mapping], Mapping]]:
    nlimit = config.get('nlimit', 0)
    t0 = config.get('t0', 0)
    transform_t = lambda t: t - t0
    return pipe(concatv(config.get('image_transformers_of_each_hit', tuple()), repeat({})),
                map(lambda other: merge(config.get('image_transformer', {}), other)),
                map(call_with_kwargs(affine_transform)),  # transform_xy
                map(transformer(transform_t)),
                take(nlimit), tuple)
transformers_with_keys = {k: transformers(config[k]) for k in ('ion', 'electron')}


@curry
def master_filter(filters: Mapping, event: Mapping) -> bool:
    for k, f in {k: is_between(fr, to) for k, (fr, to) in filters}.items():
        if k not in event:
            return False
        if not f(event[k]):
            return False
    return True


@curry
def accelerator(is_master: Callable[[Mapping], bool], momentum_xy: Callable[[T, T, T], Sequence[T]],
                momentum_z: Callable[[T], T], event: Mapping) -> Mapping:
    if not is_master(event):
        return event
    px, py = momentum_xy(event['x'], event['y'], event['t'])
    pz = momentum_z(event['t'])
    return merge(event, {'px': px, 'py': py, 'pz': pz})


def accelerators(config: Mapping) -> Sequence[Callable[[Mapping], Mapping]]:
    master = pipe(config.get('master_filters_of_each_hit', tuple()), map(event_filter), tuple)
    particles = config.get('particles_of_each_hit', tuple())
    xy = pipe(particles,
              map(lambda kwargs: momentum_xy(**{'magnetic_filed': config.get('magnetic_filed', 0)}, **kwargs)), tuple)
    acc = pipe(config.get('accelerators_of_each_region', tuple()), map(call_with_kwargs(accelerate)), tuple)
    z = pipe(particles, map(lambda p: pipe(tuple(a(**p) for a in acc), compose_accelerators, momentum_z)), tuple)
    return pipe(zip(master, xy, z), map(call_with_args(accelerator)), tuple)
accelerators_with_keys = {k: accelerators(config[k]) for k in ('ion', 'electron')}


with Pool(8) as p:
    pmap = curry(p.imap)
    ret = pipe(config['filenames'], map(read(config['treename'])), concat, take(100),
               pmap(lambda other: merge_with(compose(tuple, map(call_with_args(call)), call_with_args(zip)),
                                             transformers_with_keys, other)),
               pmap(lambda other: merge_with(compose(tuple, call_with_args(filter)), global_filter_with_keys, other)),
               pmap(lambda other: merge_with(compose(tuple, map(call_with_args(call)), call_with_args(zip)),
                                             accelerators_with_keys, other)),
               map(lambda arg: print(arg['electron'])), tuple)
