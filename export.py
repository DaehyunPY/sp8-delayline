import json
from collections import deque
from glob import iglob as glob
from itertools import repeat, count
from multiprocessing import Pool
from os import chdir
from os.path import abspath, dirname
from sys import argv
from typing import Mapping, Iterable, Sequence, Optional, Callable

import yaml
from cytoolz import concat, pipe, curry, take, compose, unique, merge, partition_all
from cytoolz.curried import take, map, merge, filter, flip
from numpy import arange
from pandas import DataFrame
from toolz.sandbox import unzip
from tqdm import tqdm

from anatools import Read, is_between, affine_transform, accelerator, Momentum, Object, Objects, with_unit


def formated_config(config: Mapping) -> Mapping:
    filenames = pipe(config['events']['filenames'], map(glob), concat, unique, sorted, tuple)
    ion_t0 = with_unit(config['ion']['time_zero_of_flight_time'])
    electron_t0 = with_unit(config['electron']['time_zero_of_flight_time'])
    spectrometer = {k: with_unit(v) for k, v in config['spectrometer'].items()}
    ions = tuple({k: with_unit(v) for k, v in ion.items()} for ion in config['ions'])
    electrons = config['electrons']['number_of_hits']
    return {
        'filenames': filenames,
        'treename': config['events']['treename'],
        'prefix': config.get('prefix', ''),
        'processes': config.get('processes', 1),
        'ion': {
            'global_filter': {
                't': (ion_t0, ion_t0 + with_unit(config['ion']['dead_time'])),
                'flag': (0, config['ion']['dead_flag'])},
            'image_transformer': {
                'th': with_unit(config['ion']['axis_angle_of_detector']),
                'x0': with_unit(config['ion']['x_zero_of_image']),
                'y0': with_unit(config['ion']['y_zero_of_image']),
                'dx': config['ion']['pixel_size_of_x'],
                'dy': config['ion']['pixel_size_of_y']},
            'image_transformers_of_each_hit': tuple(({
                'x1': ion['x_shift'],
                'y1': ion['y_shift']} for ion in ions)),
            't0': ion_t0,
            'master_filters_of_each_hit': tuple(({
                't': (ion['flight_time_from'],
                      ion['flight_time_to'])} for ion in ions)),
            'accelerators_of_each_region': tuple(({
                                                      'electric_filed': (
                                                          (spectrometer['electric_potential_of_Electron'] -
                                                           spectrometer['electric_potential_of_Ion1nd']) /
                                                          (spectrometer['length_of_LReg'] +
                                                           spectrometer['length_of_DReg'])),
                                                      'length': spectrometer['length_of_LReg']}, {
                                                      'electric_filed': (
                                                          (spectrometer['electric_potential_of_Ion1nd'] -
                                                           spectrometer['electric_potential_of_Ion2nd']) /
                                                          spectrometer['length_of_AccReg']),
                                                      'length': spectrometer['length_of_AccReg']}, {
                                                      'electric_filed': (
                                                          (spectrometer['electric_potential_of_Ion2nd'] -
                                                           spectrometer['electric_potential_of_IonMCP']) /
                                                          spectrometer['length_of_GepReg']),
                                                      'length': spectrometer['length_of_GepReg']})),
            'magnetic_filed': spectrometer['uniform_magnetic_field'],
            'particles_of_each_hit': tuple(({
                'mass': ion['mass'],
                'charge': ion['charge']} for ion in ions))},
        'electron': {
            'global_filter': {
                't': (electron_t0, electron_t0 + with_unit(config['electron']['dead_time'])),
                'flag': (0, config['electron']['dead_flag'])},
            'image_transformer': {
                'th': with_unit(config['electron']['axis_angle_of_detector']),
                'x0': with_unit(config['electron']['x_zero_of_image']),
                'y0': with_unit(config['electron']['y_zero_of_image']),
                'dx': config['electron']['pixel_size_of_x'],
                'dy': config['electron']['pixel_size_of_y']},
            't0': electron_t0,
            'master_filters_of_each_hit': ({
                                               't': (with_unit(config['electrons']['flight_time_from']),
                                                     with_unit(config['electrons']['flight_time_to']))},) * electrons,
            'accelerators_of_each_region': tuple(({
                                                      'electric_filed': (
                                                          (spectrometer['electric_potential_of_Ion1nd'] -
                                                           spectrometer['electric_potential_of_Electron']) /
                                                          (spectrometer['length_of_LReg'] +
                                                           spectrometer['length_of_DReg'])),
                                                      'length': spectrometer['length_of_DReg']}, {
                                                      'electric_filed': 0,
                                                      'length': spectrometer['length_of_DraftReg']})),
            'magnetic_filed': spectrometer['uniform_magnetic_field'],
            'particles_of_each_hit': ({
                                          'mass': 1,
                                          'charge': -1},) * electrons}}


@curry
def read(treename: str, filename: str) -> Iterable[Mapping]:
    with Read(filename) as r:
        print("Reading root file: '{}'...".format(filename))
        yield from tqdm(r[treename])


@curry
def read_sliced(treename: str, filename: str, slice: slice):
    with Read(filename) as r:
        return r[treename][slice]


@curry
def fold(treename: str, filename: str, map=map, chunksize=128):  # todo: suppose to work, but not working with 'imap'
    with Read(filename) as r:
        print("Reading root file: '{}'...".format(filename))
        n = arange(len(r[treename]))
    chunks = partition_all(chunksize, n)
    slices = (slice(chunk[0], chunk[-1] + 1) for chunk in chunks)
    return tqdm(concat(map(read_sliced(treename, filename), slices)), total=len(n))


@curry
def keys_are_between(filters: Mapping, event: Mapping) -> bool:
    for k, b in {k: is_between(fr, to) for k, (fr, to) in filters.items()}.items():
        if k not in event:
            return False
        if not b(event[k]):
            return False
    return True


class Analysis:
    def __init__(self, global_filter: Mapping = {}, t0: float = 0, image_transformer: Mapping = {},
                 image_transformers_of_each_hit: Sequence[Mapping] = (),
                 master_filters_of_each_hit: Sequence[Mapping] = (),
                 particles_of_each_hit: Sequence[Mapping] = (),
                 accelerators_of_each_region: Sequence[Mapping] = (), magnetic_filed: float = 0):
        self.__global_filter = pipe(global_filter, keys_are_between, filter)

        self.__t0 = t0
        nlimit = len(master_filters_of_each_hit)
        each_hits = pipe(concat((image_transformers_of_each_hit, repeat({}))), take(nlimit))
        merged = (merge(image_transformer, d) for d in each_hits)
        self.__image_transformers = tuple(affine_transform(**d) for d in merged)

        self.__master_conditions: Sequence[Callable[[Mapping], bool]] = tuple(keys_are_between(d)
                                                                              for d in master_filters_of_each_hit)
        acc = accelerator(*accelerators_of_each_region)
        self.__calculators = tuple(Momentum(accelerator=acc, magnetic_filed=magnetic_filed, **p)
                                   for p in particles_of_each_hit)

    def __hit_transform(self, event):
        for transformer, hit in zip(self.__image_transformers, event):
            x, y = transformer(hit['x'], hit['y'])
            yield {'x': x, 'y': y, 't': hit['t'] - self.__t0}

    def __is_in_master(self, event):
        if len(self.__master_conditions) > len(event):
            return False
        for condition, hit in zip(self.__master_conditions, event):
            if not condition(hit):
                return False
        return True

    def __hit_calculate(self, event):
        for calculator, hit in zip(self.__calculators, event):
            ke, px, py, pz = calculator(hit['x'], hit['y'], hit['t'])
            yield {**hit, **{'ke': ke, 'px': px, 'py': py, 'pz': pz}}

    def __call__(self, event: Mapping) -> Optional[Objects]:
        filtered = self.__global_filter(event)
        transformed = tuple(self.__hit_transform(filtered))
        if not self.__is_in_master(transformed):
            return None
        calculated = self.__hit_calculate(transformed)
        return Objects(*(Object(**d) for d in calculated))


def export(ion_events: Iterable[Objects], electron_events: Iterable[Objects]) -> Iterable[Mapping]:
    for ions, electrons in zip(ion_events, electron_events):
        if (ions is not None) and (electrons is not None):
            yield pipe(concat((({'i{}h_t'.format(n): i.t,
                                 'i{}h_x'.format(n): i.x,
                                 'i{}h_y'.format(n): i.y,
                                 'i{}h_ke'.format(n): i.ke,
                                 'i{}h_px'.format(n): i.px,
                                 'i{}h_py'.format(n): i.py,
                                 'i{}h_pz'.format(n): i.pz} for n, i in zip(count(1), ions)),
                               ({'e{}h_t'.format(n): e.t,
                                 'e{}h_x'.format(n): e.x,
                                 'e{}h_y'.format(n): e.y,
                                 'e{}h_ke'.format(n): e.ke,
                                 'e{}h_px'.format(n): e.px,
                                 'e{}h_py'.format(n): e.py,
                                 'e{}h_pz'.format(n): e.pz} for n, e in zip(count(1), electrons)))),
                       merge)


if __name__ == '__main__':
    if len(argv) == 1:
        raise ValueError("Usage: program config")
    elif len(argv) == 2:
        config_ = abspath(argv[1])
        chdir(dirname(config_))
    else:
        raise ValueError("Too many arguments!: '{}'".format(argv[1:]))

    with open(config_, 'r') as f:
        print("Reading config file: '{}'...".format(config_))
        config = formated_config(yaml.load(f))
    prefix = config['prefix']

    with open('{}loaded_config.json'.format(prefix), 'w') as f:
        json.dump(config, f, sort_keys=True, indent=2)

    ion_analysis = Analysis(**config['ion'])
    electron_analysis = Analysis(**config['electron'])


    def analysis(event):
        return ion_analysis(event['ion']), electron_analysis(event['electron'])

    events = pipe(config['filenames'], map(read(config['treename'])), concat)
    execute = flip(deque, 0)

    with Pool(config.get('processes', 1)) as p:
        pmap = curry(p.imap_unordered)
        ion_events, electron_events = pipe(events, pmap(analysis), unzip)

        with open('{}exported.json'.format(prefix), 'w') as f:
            write = map(compose(f.write, '{}\n'.format, json.dumps))
            pipe(export(ion_events, electron_events), write, execute)

    with open('{}exported.json'.format(prefix), 'r') as f:
        df = pipe(f, map(json.loads), list, DataFrame)
    df.to_csv('{}exported.csv'.format(prefix))
