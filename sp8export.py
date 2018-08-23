#!/usr/bin/env python3
from functools import reduce
from glob import iglob
from itertools import islice, chain
from math import sin, cos
from typing import List
from sys import argv

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import udf, array, size
from yaml import safe_load

from dltools import (in_degree, in_milli_meter, in_electron_volt, in_gauss, in_atomic_mass, Hit, SpkHits,
                     uniform_electric_field, none_field, ion_spectrometer, electron_spectrometer)


if __name__ == '__main__':
    # %% read config
    if len(argv) == 1:
        raise ValueError("Usage: program config")
    elif 2 < len(argv):
        raise ValueError("Too many arguments!: '{}'".format(argv[1:]))
    with open(argv[1], 'r') as f:
        config = safe_load(f)

    target_files = config['target_files']
    save_as = config.get('save_as', 'exported.parquet')
    nparts = config.get('repartiton', None)
    c_spk = config['spark']
    c_spt = config['spectrometer']
    c_ion = config['ion']
    c_ionp = config['ion_momemtum_calculator']
    c_ele = config['electron']
    c_elep = config['electron_momemtum_calculator']


    # %% initialize spark builder
    builder = (SparkSession
               .builder
               .appName("sp8export")
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
        k: ion_spectrometer(ion_acc, mass=in_atomic_mass(d['mass']), charge=d['charge'], safe_pz_range=d['safe_pz_range'])
        for k, d in c_ionp.items()
    }
    ele_spt = electron_spectrometer(ele_acc, magnetic_filed=in_gauss(c_spt['uniform_mfield']))


    # %% spark udfs
    @udf(SpkHits)
    def combine_hits(tarr: List[float],
                     xarr: List[float],
                     yarr: List[float],
                     flagarr: List[float],  # todo: nullable
                     nhits: int,  # todo: nullable
                     ) -> List[dict]:
        zipped = ({'t': t, 'x': x, 'y': y, 'flag': f} for t, x, y, f in zip(tarr, xarr, yarr, flagarr))
        return list(islice(zipped, nhits))


    @udf(SpkHits)
    def analyse_ihits(hits: List[dict]) -> List[dict]:
        notdead = ({'t': h['t'] - c_ion['t0'],
                    'x': c_ion['dx'] * (h['x'] - c_ion['x0']),
                    'y': c_ion['dy'] * (h['y'] - c_ion['y0']),
                    'as': {},
                    } for h in hits
                   if (not c_ion['dead_flag'] < h['flag']) and (0 < (h['t'] - c_ion['t0']) < c_ion['dead_time']))
        for h in notdead:
            for k, d in c_ionp.items():
                if d['fr'] < h['t'] < d['to']:
                    h['as'].update(**{k: ion_spt[k](Hit.in_experimental_units(t=h['t'],
                                                                              x=h['x'] - d['x1'],
                                                                              y=h['y'] - d['y1']))
                                   .to_experimental_units()})
        return notdead


    @udf(SpkHits)
    def analyse_ehits(hits: List[dict]) -> List[dict]:
        th = in_degree(c_ele['th'])
        notdead = ({'t': h['t'] - c_ele['t0'],
                    'x': c_ele['dx'] * (cos(th) * h['x'] - sin(th) * h['y'] - c_ele['x0']),
                    'y': c_ele['dy'] * (sin(th) * h['x'] + cos(th) * h['y'] - c_ele['y0']),
                    'as': {},
                    } for h in hits
                   if (not c_ele['dead_flag'] < h['flag']) and (0 < (h['t'] - c_ele['t0']) < c_ele['dead_time']))
        for h in notdead:
            if c_elep['fr'] < h['t'] < c_elep['to']:
                h['as'].update(**{'e': ele_spt(Hit.in_experimental_units(t=h['t'], x=h['x'], y=h['y']))
                               .to_experimental_units()})
        return notdead


    # %% connect to spark master & read root files
    with builder.getOrCreate() as spark:
        globbed = chain.from_iterable(iglob(f) for f in target_files)
        loaded = (spark.read.format("org.dianahep.sparkroot").load(fn) for fn in globbed)
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
        if nparts is None:
            repartitioned = analyzed
        else:
            repartitioned = analyzed.repartition(nparts)
        (repartitioned
         .write
         .parquet(save_as)
         )
