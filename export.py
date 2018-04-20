from collections import ChainMap
from glob import glob
from itertools import count, chain
from os import chdir
from os.path import abspath, dirname
from sys import argv
from typing import Sequence

from cytoolz import concat, pipe, unique, partial, map, filter
from dask.bag import from_sequence
from dask.diagnostics import ProgressBar
from dask.multiprocessing import get as multiprocessing_get
from numba import jit
from numpy import nan
from yaml import load as load_yaml

from sp8tools import (call_with_kwargs, affine_transform, queries, events, with_unit,
                      as_milli_meter, as_nano_sec, as_electron_volt, uniform_electric_field, none_field,
                      ion_spectrometer, electron_spectrometer, Hit)


def load_config(config: dict) -> None:
    global filenames, treename, chunk_size, prefix
    filenames = pipe(config['events']['filenames'], partial(map, glob), concat, unique, sorted)
    treename = config['events']['treename']
    chunk_size = config.get('chunk_size', 1000000)
    prefix = config.get('prefix', '')

    global ion_t0, ion_hit, ion_nhits, electron_t0, electron_hit, electron_nhits
    ion_t0 = with_unit(config['ion']['time_zero_of_flight_time'])
    ion_hit = {
        't': [ion_t0, ion_t0 + with_unit(config['ion']['dead_time'])],
        'flag': [0, config['ion']['dead_flag']]
    }
    ion_nhits = len(config['ions'])
    electron_t0 = with_unit(config['electron']['time_zero_of_flight_time'])
    electron_hit = {
        't': [electron_t0, electron_t0 + with_unit(config['electron']['dead_time'])],
        'flag': [0, config['electron']['dead_flag']]
    }
    electron_nhits = config['electrons']['number_of_hits']

    global ion_imgs, electron_imgs
    ions = [{k: with_unit(v) for k, v in ion.items()} for ion in config['ions']]
    ion_img = {
        'th': with_unit(config['ion']['axis_angle_of_detector']),
        'x0': with_unit(config['ion']['x_zero_of_image']),
        'y0': with_unit(config['ion']['y_zero_of_image']),
        'dx': config['ion']['pixel_size_of_x'],
        'dy': config['ion']['pixel_size_of_y'],
        'x1': 0,
        'y1': 0
    }
    ion_imgs = [
        ChainMap({'x1': ion.get('x_shift', 0), 'y1': ion.get('y_shift', 0)}, ion_img) for ion in ions
    ]
    electron_img = {
        'th': with_unit(config['electron']['axis_angle_of_detector']),
        'x0': with_unit(config['electron']['x_zero_of_image']),
        'y0': with_unit(config['electron']['y_zero_of_image']),
        'dx': config['electron']['pixel_size_of_x'],
        'dy': config['electron']['pixel_size_of_y'],
        'x1': 0,
        'y1': 0
    }
    electron_imgs = [electron_img] * electron_nhits

    global ion_master, electron_master
    ion_master = [
        [ion['flight_time_from'], ion['flight_time_to']] for ion in ions
    ]
    electron_master = [
        [with_unit(config['electrons']['flight_time_from']), with_unit(config['electrons']['flight_time_to'])]
    ] * electron_nhits

    global ion_calculators, electron_calculators
    spectrometer = {k: with_unit(v) for k, v in config['spectrometer'].items()}
    ion_acc = (
            uniform_electric_field(length=spectrometer['length_of_GepReg'],
                                   electric_field=(
                                           (spectrometer['electric_potential_of_Ion2nd'] -
                                            spectrometer['electric_potential_of_IonMCP']) /
                                           spectrometer['length_of_GepReg'])) *
            uniform_electric_field(length=spectrometer['length_of_AccReg'],
                                   electric_field=(
                                                 (spectrometer['electric_potential_of_Ion1nd'] -
                                                  spectrometer['electric_potential_of_Ion2nd']) /
                                                 spectrometer['length_of_AccReg'])) *
            uniform_electric_field(length=spectrometer['length_of_LReg'],
                                   electric_field=(
                                           (spectrometer['electric_potential_of_Electron'] -
                                            spectrometer['electric_potential_of_Ion1nd']) /
                                           (spectrometer['length_of_LReg'] +
                                            spectrometer['length_of_DReg']))))
    ion_calculators = [
        ion_spectrometer(ion_acc, mass=ion['mass'], charge=ion['charge'], safe_pz_range=ion.get('safe_pz_range', 400))
        if 'mass' in ion else None
        for ion in ions
    ]
    ele_acc = (
            none_field(length=spectrometer['length_of_DraftReg']) *
            uniform_electric_field(length=spectrometer['length_of_DReg'],
                                   electric_field=(
                                           (spectrometer['electric_potential_of_Ion1nd'] -
                                            spectrometer['electric_potential_of_Electron']) /
                                           (spectrometer['length_of_LReg'] +
                                            spectrometer['length_of_DReg']))))
    electron_calculators = [
        electron_spectrometer(ele_acc, magnetic_filed=spectrometer['uniform_magnetic_field'])
    ] * electron_nhits


@jit
def ion_hit_filter(hit: dict) -> bool:
    for k, (fr, to) in ion_hit.items():
        if not fr <= hit[k]:
            return False
        if not hit[k] <= to:
            return False
    return True


@jit
def electron_hit_filter(hit: dict) -> bool:
    for k, (fr, to) in electron_hit.items():
        if not fr <= hit[k]:
            return False
        if not hit[k] <= to:
            return False
    return True


@call_with_kwargs
@jit
def hit_filter(ions: Sequence[dict], electrons: Sequence[dict]) -> dict:
    filtered_ions = list(filter(ion_hit_filter, ions))
    filtered_electrons = list(filter(electron_hit_filter, electrons))
    return {
        'inhits': len(filtered_ions),
        'enhits': len(filtered_electrons),
        'ions': filtered_ions,
        'electrons': filtered_electrons
    }


@call_with_kwargs
@jit
def nhits_filter(inhits: int, enhits: int, ions: Sequence[dict], electrons: Sequence[dict]) -> bool:
    return ion_nhits <= inhits and electron_nhits <= enhits


@call_with_kwargs
def hit_transformer(inhits: int, enhits: int, ions: Sequence[dict], electrons: Sequence[dict]) -> dict:
    return {
        'ions': [
            {'t': h['t'] - ion_t0, **dict(zip(['x', 'y'], [*affine_transform(h['x'], h['y'], **img)]))}
            for img, h in zip(ion_imgs, ions)
        ],
        'electrons': [
            {'t': h['t'] - electron_t0, **dict(zip(['x', 'y'], [*affine_transform(h['x'], h['y'], **img)]))}
            for img, h in zip(electron_imgs, electrons)
        ]
    }


@call_with_kwargs
def master_filter(ions: Sequence[dict], electrons: Sequence[dict]) -> bool:
    for h, (fr, to) in zip(ions, ion_master):
        t = h['t']
        if not (fr <= t) & (t <= to):
            return False
    for h, (fr, to) in zip(electrons, electron_master):
        t = h['t']
        if not (fr <= t) & (t <= to):
            return False
    return True


@call_with_kwargs
def hit_calculator(ions: Sequence[dict], electrons: Sequence[dict]) -> dict:
    return {
        'ions': [
            {**h, **f(Hit(**h))._asdict()}
            if f is not None else {**h, **dict(zip(['ke', 'px', 'py', 'pz'], (nan, nan, nan, nan)))}
            for h, f in zip(ions, ion_calculators)],
        'electrons': [
            {**h, **f(Hit(**h))._asdict()}
            for h, f in zip(electrons, electron_calculators)
        ]
    }


@jit
def as_the_units(x: float, y: float, t: float, ke: float, px: float, py: float, pz: float) -> dict:
    return {
        'x': as_milli_meter(x),
        'y': as_milli_meter(y),
        't': as_nano_sec(t),
        'ke': as_electron_volt(ke),
        'px': px,
        'py': py,
        'pz': pz
    }


@call_with_kwargs
def unit_mapper(ions: Sequence[dict], electrons: Sequence[dict]) -> dict:
    return {
        'ions': [as_the_units(**h) for h in ions],
        'electrons': [as_the_units(**h) for h in electrons]
    }


@call_with_kwargs
def flat_event(ions: Sequence[dict], electrons: Sequence[dict]) -> dict:
    iformat = 'i{}h_{}'.format
    eformat = 'e{}h_{}'.format
    items = dict.items
    ihits = (
        [[iformat(i, k), v] for k, v in items(h) if k in {'x', 'y', 't', 'ke', 'px', 'py', 'pz'}]
        for i, h in zip(count(1), ions)
    )
    ehits = (
        [[eformat(i, k), v] for k, v in items(h) if k in {'x', 'y', 't', 'ke', 'px', 'py', 'pz'}]
        for i, h in zip(count(1), electrons)
    )
    return dict(chain(*chain(ihits, ehits)))


@call_with_kwargs
def event_list(*args, **kwargs):
    return events(*args, **kwargs)


if __name__ == '__main__':
    if len(argv) == 1:
        raise ValueError("Usage: program config")
    elif len(argv) == 2:
        config_filename = abspath(argv[1])
        working_directory = dirname(config_filename)
        print("Change working directory to '{}'...".format(working_directory))
        chdir(working_directory)
    else:
        raise ValueError("Too many arguments!: '{}'".format(argv[1:]))

    with open(config_filename, 'r') as f:
        print("Reading config file: '{}'...".format(config_filename))
        config = load_yaml(f)
    load_config(config)

    que = [*chain(*(queries(filename, treename, chunk_size) for filename in filenames))]
    print("Files:")
    for filename in filenames:
        print('    {}'.format(filename))
    print('    Total {} Files'.format(len(filenames)))
    print("Chunk Size: {}".format(chunk_size))
    print("Number of ROOT Partitions: {}".format(len(que)))

    # from tqdm import tqdm
    # calculated = pipe(
    #     que,
    #     partial(map, event_list),
    #     concat,
    #     # tqdm,
    #     partial(map, hit_filter),
    #     partial(filter, nhits_filter),
    #     partial(map, hit_transformer),
    #     partial(filter, master_filter),
    #     partial(map, hit_calculator),
    #     partial(map, unit_mapper),
    #     # tqdm
    # )
    # for hit in calculated:
    #     continue

    whole_events = from_sequence(que).map(event_list).flatten()
    calculated = (
        whole_events
            .map(hit_filter)
            .filter(nhits_filter)
            .map(hit_transformer)
            .filter(master_filter)
            .map(hit_calculator)
            .map(unit_mapper)
    )
    with ProgressBar():
        calculated.map(flat_event).to_dataframe().to_csv('{}exported-*.csv'.format(prefix), get=multiprocessing_get)
        # from json import dumps as dump_json
        # calculated.map(dump_json).to_textfiles('{}exported-*.json'.format(prefix))
