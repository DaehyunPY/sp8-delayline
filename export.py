from glob import iglob as glob
from itertools import repeat, count
from json import dump
from operator import sub, getitem
from textwrap import dedent
from typing import TypeVar, Sequence, Mapping, Callable, Iterable, Tuple

from cytoolz import concat, concatv, pipe, curry, take, juxt, compose, unique, merge
from cytoolz.curried import take, map, merge, filter, flip
from dill import loads, dumps
from numba import jit
from pandas import DataFrame
from pathos.multiprocessing import Pool
from toolz.sandbox import unzip
from yaml import load

from anatools import (Read, is_between, affine_transform, accelerate, compose_accelerators, momentum, Object,
                      Objects, as_atomic_mass, as_nano_sec, with_unit)

T = TypeVar('T')


def formated(config):
    filenames = pipe(config['events']['filenames'], map(glob), concat, unique, sorted, tuple)
    ion_t0 = with_unit(config['ion']['time_zero_of_flight_time'])
    electron_t0 = with_unit(config['electron']['time_zero_of_flight_time'])
    spectrometer = {k: with_unit(v) for k, v in config['spectrometer'].items()}
    ions = tuple({k: with_unit(v) for k, v in ion.items()} for ion in config['ions'])
    electrons = config['electrons']['number_of_hits']
    return {
        'filenames': filenames,
        'treename': config['events']['treename'],
        'ion': {
            'global_filter': {
                't': (ion_t0, ion_t0 + with_unit(config['ion']['dead_time'])),
                'flag': (0, config['ion']['dead_flag'])},
            'nlimit': config['events']['limit_of_hits'],
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
            'nlimit': config['events']['limit_of_hits'],
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
        yield from r[treename]


@curry
def event_filter(filters: Mapping, event: Mapping) -> bool:
    for k, b in {k: is_between(fr, to) for k, (fr, to) in filters.items()}.items():
        if k not in event:
            return False
        if not b(event[k]):
            return False
    return True


@curry
@jit
def transformer(t: Callable[[T], T], xy: Callable[[T, T], Sequence[T]], event: Mapping) -> Mapping:
    x, y = xy(event['x'], event['y'])
    return merge(event, {'x': x, 'y': y, 't': t(event['t'])})


@curry
@jit
def accelerator(master: Callable[[Mapping], bool], p: Callable[[T, T, T], Tuple[T, T, T, T]],
                event: Mapping) -> Mapping:
    if not master(event):
        return event
    ke, px, py, pz = p(event['x'], event['y'], event['t'])
    return merge(event, {'ke': ke, 'px': px, 'py': py, 'pz': pz})


def processor(m: Mapping) -> Callable[[Mapping], Objects]:
    filters = pipe(m.get('global_filter', {}), event_filter, dumps)

    nlimit = m.get('nlimit', 0)
    t = flip(sub, m.get('t0', 0))
    merged = (merge(m.get('image_transformer', {}), m)
              for m in concatv(m.get('image_transformers_of_each_hit', tuple()), repeat({})))
    xy = (affine_transform(**d) for d in merged)
    transformers = pipe((transformer(t, xy_) for xy_ in xy), take(nlimit), tuple, dumps)

    master = (event_filter(filters) for filters in m.get('master_filters_of_each_hit', tuple()))
    particles = m.get('particles_of_each_hit', tuple())
    acc = compose_accelerators(accelerate(**reg) for reg in m.get('accelerators_of_each_region', tuple()))
    for i, p in zip(count(), particles):
        mass = p['mass']
        charge = p['charge']
        t = acc(mass, charge, 0)['flight_time']
        print(dedent("""\
                     object #{}:
                        mass (u): {mass}
                        charge (au): {charge}
                        TOF at pz=0 (ns): {t}\
                     """.format(i, mass=as_atomic_mass(mass), charge=charge, t=as_nano_sec(t))))
    mmt = (momentum(**p, accelerator=acc, magnetic_filed=m.get('magnetic_filed', 0)) for p in particles)
    accelerators = pipe((accelerator(m, p) for m, p in zip(master, mmt)), tuple, dumps)

    def process(event: Mapping) -> Objects:
        filtered = filter(loads(filters), event)
        transformed = (f(e) for f, e in zip(loads(transformers), filtered))
        accelerated = (f(e) for f, e in zip(loads(accelerators), transformed))
        return Objects(*(Object(**d) for d in accelerated))

    return process


def export(ion_events: Iterable[Objects], electron_events: Iterable[Objects]):
    for ions, electrons in zip(ion_events, electron_events):
        if not ((len(ions.having_momentum) == 0) and (len(electrons.having_momentum) == 0)):
            yield pipe(concatv(({'i{}h_t'.format(n): i.t,
                                 'i{}h_x'.format(n): i.x,
                                 'i{}h_y'.format(n): i.y,
                                 'i{}h_ke'.format(n): i.ke,
                                 'i{}h_px'.format(n): i.px,
                                 'i{}h_py'.format(n): i.py,
                                 'i{}h_pz'.format(n): i.pz} for n, i in zip(count(1), ions.having_momentum)),
                               ({'e{}h_t'.format(n): e.t,
                                 'e{}h_x'.format(n): e.x,
                                 'e{}h_y'.format(n): e.y,
                                 'e{}h_ke'.format(n): e.ke,
                                 'e{}h_px'.format(n): e.px,
                                 'e{}h_py'.format(n): e.py,
                                 'e{}h_pz'.format(n): e.pz} for n, e in zip(count(1), electrons.having_momentum))),
                       merge)


with open('config.yaml', 'r') as f:
    config = formated(load(f))
with open('loaded_config.json', 'w') as f:
    dump(config, f, sort_keys=True, indent=2)

ion_processor = compose(processor(config['ion']), flip(getitem, 'ion'))
electron_processor = compose(processor(config['electron']), flip(getitem, 'electron'))
events = pipe(config['filenames'], map(read(config['treename'])), concat)

with Pool(16) as p:
    pmap = curry(p.imap)
    ion_events, electron_events = pipe(events, pmap(juxt(ion_processor, electron_processor)), unzip)
    df = pipe(export(ion_events, electron_events), list, DataFrame)
df.to_csv('exported.csv')
