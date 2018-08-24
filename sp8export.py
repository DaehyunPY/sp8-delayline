#!/usr/bin/env python3
# %% external dependencies
from os import chdir
from os.path import dirname
from typing import List
from glob import iglob
from math import sin, cos, nan
from argparse import ArgumentParser
from functools import reduce, partial
from itertools import islice, chain

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import udf, array, size
from yaml import safe_load

from dltools import (in_degree, in_milli_meter, in_electron_volt, in_gauss, in_atomic_mass, Hit, SpkHits,
                     uniform_electric_field, none_field, ion_spectrometer, electron_spectrometer)


# %% parser & parameters
parser = ArgumentParser(prog='sp8export', description="Analyse SP8 resorted data and save as parquet format.")
parser.add_argument('config', type=str, default='config.yaml', nargs='?',
                    help='config filename')

if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = safe_load(f)
    chdir(dirname(args.config))
    target_files = config.get('target_files', ['*.root'])
    save_as = config.get('save_as', 'exported.parquet')
    c_spk = config.get('spark', {})
    c_spt = config['spectrometer']
    c_ions = config['ions']
    c_ionp = config['ion_momemtum_calculator']
    c_eles = config['electrons']
    c_elep = config['electron_momemtum_calculator']

    # %% initialize spark builder
    builder = (SparkSession
               .builder
               .config("spark.jars.packages", "org.diana-hep:spark-root_2.11:0.1.15")
               )
    for k, v in c_spk.items():
        builder.config(k, v)

    # %% initialize spectrometers
    ion_acc = (
            uniform_electric_field(length=in_milli_meter(c_spt['mcpgep_reg']),
                                   electric_field=(in_electron_volt(c_spt['ion2nd_epoten'] - c_spt['ionmcp_epoten'])
                                                   / in_milli_meter(c_spt['mcpgep_reg']))) *
            uniform_electric_field(length=in_milli_meter(c_spt['acc_reg']),
                                   electric_field=(in_electron_volt(c_spt['ion1st_epoten'] - c_spt['ion2nd_epoten'])
                                                   / in_milli_meter(c_spt['acc_reg']))) *
            uniform_electric_field(length=in_milli_meter(c_spt['ionsep_reg']),
                                   electric_field=(in_electron_volt(c_spt['electron_epoten'] - c_spt['ion1st_epoten'])
                                                   / in_milli_meter(c_spt['ionsep_reg'] + c_spt['elesep_reg'])))
    )
    ele_acc = (
            none_field(length=in_milli_meter(c_spt['draft_reg'])) *
            uniform_electric_field(length=in_milli_meter(c_spt['elesep_reg']),
                                   electric_field=(in_electron_volt(c_spt['ion1st_epoten'] - c_spt['electron_epoten'])
                                                   / in_milli_meter(c_spt['ionsep_reg'] + c_spt['elesep_reg'])))
    )
    ion_spt = {
        k: {'fr': d['fr'],
            'to': d['to'],
            'x1': d.get('x1', 0),
            'y1': d.get('y1', 0),
            'f': ion_spectrometer(ion_acc,
                                  mass=in_atomic_mass(d['mass']),
                                  charge=d['charge'],
                                  safe_pz_range=d['safe_pz_range'])}
        for k, d in c_ionp.items()
    }
    ele_spt = {
        'e': {'fr': c_elep['fr'],
              'to': c_elep['to'],
              'x1': c_elep.get('x1', 0),
              'y1': c_elep.get('y1', 0),
              'f': electron_spectrometer(ele_acc,
                                         magnetic_filed=in_gauss(c_spt['uniform_mfield']),
                                         safe_pz_range=c_elep['safe_pz_range'])}
    }

    # %% functions
    @udf(SpkHits)
    def combine_hits(tarr: List[float],
                     xarr: List[float],
                     yarr: List[float],
                     flagarr: List[float],  # todo: nullable
                     nhits: int,  # todo: nullable
                     ) -> List[dict]:
        zipped = ({'t': t, 'x': x, 'y': y, 'flag': f} for t, x, y, f in zip(tarr, xarr, yarr, flagarr))
        return list(islice(zipped, nhits))

    def analyse_hits(hits: List[dict],
                     th=0, t0=0, x0=0, y0=0, dx=1, dy=1, x1=0, y1=0, dead_time=nan,
                     targets: dict = None) -> List[dict]:
        """
        :param targets: Momentum calculator. Example:
            targets = {
                'C_1': {
                    'fr': 300,  # ns
                    'to':  1000,  # ns
                    'mass': 12.0107,  # u
                    'safe_pz_range': 400,  # au
                    'x1': 0,  # mm
                    'y1': 0,  # mm
                }
            }
        :return: List of analyzed hits.
        """
        thr = in_degree(th)
        notdead = [{'t': h['t'] - t0,
                    'x': dx * (cos(thr) * h['x'] - sin(thr) * h['y'] - x0),
                    'y': dy * (sin(thr) * h['x'] + cos(thr) * h['y'] - y0),
                    'flag': h['flag'],
                    'as': {},
                    } for h in hits
                   if 0 < h['t'] - t0 < dead_time]
        if targets is None:
            return notdead
        for h in notdead:
            for k, d in targets.items():
                if d['fr'] < h['t'] < d['to']:
                    h['as'][k] = (d['f'](Hit.in_experimental_units(t=h['t'],
                                                                   x=h['x'] + d.get('x1', x1),
                                                                   y=h['y'] + d.get('y1', y1)))
                                  .to_experimental_units())
        return notdead

    analyse_ihits = udf(partial(analyse_hits, targets=ion_spt, **c_ions), SpkHits)
    analyse_ehits = udf(partial(analyse_hits, targets=ele_spt, **c_eles), SpkHits)

    # %% connect to spark master & read root files
    with builder.getOrCreate() as spark:
        globbed = chain.from_iterable(iglob(patt) for patt in target_files)
        uniques = sorted(set(globbed))
        loaded = (spark.read.format("org.dianahep.sparkroot").load(f) for f in uniques)
        df = reduce(DataFrame.union, loaded)

        # %% restructure data
        imhits = sum(1 for c in df.columns if c.startswith('IonT'))
        emhits = sum(1 for c in df.columns if c.startswith('ElecT'))
        restructured = (
            df
                .withColumn('itarr', array(*['IonT{}'.format(i) for i in range(imhits)]))
                .withColumn('ixarr', array(*['IonX{}'.format(i) for i in range(imhits)]))
                .withColumn('iyarr', array(*['IonY{}'.format(i) for i in range(imhits)]))
                .withColumn('iflagarr', array(*['IonFlag{}'.format(i) for i in range(imhits)]))
                .withColumnRenamed('IonNum', 'inhits')
                .withColumn('etarr', array(*['ElecT{}'.format(i) for i in range(emhits)]))
                .withColumn('exarr', array(*['ElecX{}'.format(i) for i in range(emhits)]))
                .withColumn('eyarr', array(*['ElecY{}'.format(i) for i in range(emhits)]))
                .withColumn('eflagarr', array(*['ElecFlag{}'.format(i) for i in range(emhits)]))
                .withColumnRenamed('ElecNum', 'enhits')
                .select(combine_hits('itarr', 'ixarr', 'iyarr', 'iflagarr', 'inhits').alias('ihits'),
                        combine_hits('etarr', 'exarr', 'eyarr', 'eflagarr', 'enhits').alias('ehits'))
        )

        # %% analyse hits & export them
        analyzed = (restructured
                    .withColumn('ihits', analyse_ihits('ihits'))
                    .filter(0 < size('ihits'))
                    .withColumn('ehits', analyse_ehits('ehits'))
                    .filter(0 < size('ehits'))
                    )
        (analyzed
         .write
         .parquet(save_as)
         )
