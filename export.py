from glob import iglob as glob
from typing import Mapping, Iterable, Sequence
from itertools import count, chain
from collections import ChainMap
from operator import and_
from os import chdir
from os.path import abspath, dirname
from sys import argv

from pyspark import SparkContext, StorageLevel
from pyspark.sql import SparkSession, Row
from yaml import load as load_yaml
from cytoolz import concat, pipe, unique, partial, map, reduce
from pandas import DataFrame
from tqdm import tqdm

from anatools import (Read, call_with_kwargs, affine_transform, accelerator, Momentum,
                      with_unit, as_milli_meter, as_nano_sec, as_electron_volt)


def load_config(config: dict) -> None:
    global filenames, treename, partitions, prefix
    filenames = pipe(config['events']['filenames'], partial(map, glob), concat, unique, sorted)
    treename = config['events']['treename']
    partitions = config.get('partitions', 2**6)
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
        ChainMap({'x1': ion['x_shift'], 'y1': ion['y_shift']}, ion_img) for ion in ions
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
    ion_acc = accelerator(
        {'electric_filed': ((spectrometer['electric_potential_of_Electron'] -
                             spectrometer['electric_potential_of_Ion1nd']) /
                            (spectrometer['length_of_LReg'] +
                             spectrometer['length_of_DReg'])),
         'length': spectrometer['length_of_LReg']},
        {'electric_filed': ((spectrometer['electric_potential_of_Ion1nd'] -
                             spectrometer['electric_potential_of_Ion2nd']) /
                            spectrometer['length_of_AccReg']),
         'length': spectrometer['length_of_AccReg']},
        {'electric_filed': ((spectrometer['electric_potential_of_Ion2nd'] -
                             spectrometer['electric_potential_of_IonMCP']) /
                            spectrometer['length_of_GepReg']),
         'length': spectrometer['length_of_GepReg']}
    )
    ion_calculators = [
        Momentum(accelerator=ion_acc,
                 magnetic_filed=spectrometer['uniform_magnetic_field'],
                 mass=ion['mass'], charge=ion['charge'])
        for ion in ions
    ]
    electron_acc = accelerator(
        {'electric_filed': ((spectrometer['electric_potential_of_Ion1nd'] -
                             spectrometer['electric_potential_of_Electron']) /
                            (spectrometer['length_of_LReg'] +
                             spectrometer['length_of_DReg'])),
         'length': spectrometer['length_of_DReg']},
        {'electric_filed': 0,
         'length': spectrometer['length_of_DraftReg']}
    )
    electron_calculators = [
        Momentum(accelerator=electron_acc,
                 magnetic_filed=spectrometer['uniform_magnetic_field'],
                 mass=1, charge=-1)
    ] * electron_nhits


def read(treename: str, filename: str) -> Iterable[Mapping]:
    with Read(filename) as r:
        print("Reading root file: '{}'...".format(filename))
        yield from tqdm(r[treename])


@call_with_kwargs
def hit_filter(ions: Sequence[dict], electrons: Sequence[dict]) -> dict:
    df_ions = DataFrame(ions)
    where_ions = reduce(and_,
                        [(fr <= df_ions[k]) & (df_ions[k] <= to)
                         for k, (fr, to) in ion_hit.items()],
                        True)
    df_ions_filtered = df_ions if where_ions is True else df_ions[where_ions]
    df_electrons = DataFrame(electrons)
    where_electrons = reduce(and_,
                             [(fr <= df_electrons[k]) & (df_electrons[k] <= to)
                              for k, (fr, to) in electron_hit.items()],
                             True)
    df_electrons_filtered = df_electrons if where_electrons is True else df_electrons[where_electrons]
    return {
        'inhits': len(df_ions_filtered),
        'enhits': len(df_electrons_filtered),
        'ions': df_ions_filtered.to_dict('records'),
        'electrons': df_electrons_filtered.to_dict('records')
    }


@call_with_kwargs
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
            {**h, **dict(zip(['ke', 'px', 'py', 'pz'], f(h['x'], h['y'], h['t'])))}
            for h, f in zip(ions, ion_calculators)],
        'electrons': [
            {**h, **dict(zip(['ke', 'px', 'py', 'pz'], f(h['x'], h['y'], h['t'])))}
            for h, f in zip(electrons, electron_calculators)
        ]
    }


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
def flat_event(ions: Sequence[dict], electrons: Sequence[dict]) -> Row:
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
    return Row(**{k: float(v) for k, v in chain(*chain(ihits, ehits))})


if __name__ == '__main__':
    if len(argv) == 1:
        raise ValueError("Usage: program config")
    elif len(argv) == 2:
        config_filename = abspath(argv[1])
        chdir(dirname(config_filename))
    else:
        raise ValueError("Too many arguments!: '{}'".format(argv[1:]))

    with open(config_filename, 'r') as f:
        print("Reading config file: '{}'...".format(config_filename))
        config = load_yaml(f)
    load_config(config)

    sc = SparkContext()
    spark = SparkSession(sc)
    read_the_tree = partial(read, treename)
    storage = StorageLevel(True, True, False, False)
    whole_events = sc.parallelize(filenames).flatMap(read_the_tree)
    whole_events.persist(storage)
    partitioned = whole_events.repartition(partitions)
    flatten = (partitioned
               .map(hit_filter)
               .filter(nhits_filter)
               .map(hit_transformer)
               .filter(master_filter)
               .map(hit_calculator)
               .map(unit_mapper)
               .map(flat_event))
    flatten.persist(storage)
    df = spark.createDataFrame(flatten)
    df.write.csv('{}exported'.format(prefix), header='true', mode='overwrite')
