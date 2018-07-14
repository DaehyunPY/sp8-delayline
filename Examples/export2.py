#!/usr/bin/env python3


# %% import
from functools import reduce
from glob import iglob
from itertools import islice
from math import sin, cos
from typing import List

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import udf, array, size

from dltools import (in_degree, in_milli_meter, in_electron_volt, in_gauss, in_atomic_mass, Hit, SpkHits,
                     uniform_electric_field, none_field, ion_spectrometer, electron_spectrometer)

# %% parameters
helium = '/Volumes/analysis/{}'.format
targetfiles = helium('SPring-8/2017B/29_CH2BrI_665eV_all/resort_ahmad/ResortLess*.root')
save_as = 'exported.parquet'

# %% initialize spark builder
builder = (SparkSession
           .builder
           .appName('PySpark Example')
           .config("spark.jars.packages", "org.diana-hep:spark-root_2.11:0.1.15")
           .config("spark.cores.max", 11)
           .config("spark.executor.cores", 5)
           .config("spark.executor.memory", "4g")
           )
# spark = builder.getOrCreate()


# %% initialize spectrometers
#
#   ion2nd                 ion1st       electron
# ┌───┐│                      │             │                    │┌───┐
# │   ││                      │             │                    ││   │
# │   ││                      │             │                    ││   │
# │   ││                      │             │                    ││   │
# │ion││                      │             │                    ││ele│
# │mcp││        acc_reg       │   sep_reg   │     draft_reg      ││mcp│
# │   ││                      │             │                    ││   │
# │   ││                      │             │                    ││   │
# │   ││                      │────x────────│                    ││   │
# │   ││                      │             │                    ││   │
# └───┘│                      │             │                    │└───┘
#
#                        uniform magnetic field
#                       symbol x: reaction point
#
c = {
    'draft_reg': 67.4,  # mm
    'elesep_reg': 33,  # mm
    'ionsep_reg': 16.5,  # mm
    'acc_reg': 82.5,  # mm
    'mcpgep_reg': 10,  # mm
    'electron_epoten': -200,  # V
    'ion1st_epoten': -350,  # V
    'ion2nd_epoten': -2000,  # V
    'ionmcp_epoten': -3590,  # V
    'uniform_mfield': 6.87,  # Gauss
}
ion_acc = (
        uniform_electric_field(length=in_milli_meter(c['mcpgep_reg']),
                               electric_field=(in_electron_volt(c['ion2nd_epoten'] - c['ionmcp_epoten'])
                                               / in_milli_meter(c['mcpgep_reg']))) *
        uniform_electric_field(length=in_milli_meter(c['acc_reg']),
                               electric_field=(in_electron_volt(c['ion1st_epoten'] - c['ion2nd_epoten'])
                                               / in_milli_meter(c['acc_reg']))) *
        uniform_electric_field(length=in_milli_meter(c['ionsep_reg']),
                               electric_field=(in_electron_volt(c['electron_epoten'] - c['ion1st_epoten'])
                                               / in_milli_meter(c['ionsep_reg'] + c['elesep_reg'])))
)
ele_acc = (
        none_field(length=in_milli_meter(c['draft_reg'])) *
        uniform_electric_field(length=in_milli_meter(c['elesep_reg']),
                               electric_field=(in_electron_volt(c['ion1st_epoten'] - c['electron_epoten'])
                                               / in_milli_meter(c['ionsep_reg'] + c['elesep_reg'])))
)
ion_spt = {
    'H_1': ion_spectrometer(ion_acc, mass=in_atomic_mass(1), charge=1, safe_pz_range=200),
    'C_1': ion_spectrometer(ion_acc, mass=in_atomic_mass(12.0107), charge=1, safe_pz_range=400),
    'Br_1': ion_spectrometer(ion_acc, mass=in_atomic_mass(79.904), charge=1, safe_pz_range=400),
    'I_1': ion_spectrometer(ion_acc, mass=in_atomic_mass(126.90447), charge=1, safe_pz_range=400),
}
ele_spt = electron_spectrometer(ele_acc, magnetic_filed=in_gauss(6.87))


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
    hasgoodflag = ({'t': h['t'] - -134.6925,
                    'x': 1.22 * (h['x'] - 0.493202),
                    'y': 1.22 * (h['y'] - -1.827212),
                    'as': {},
                    } for h in hits if not 14 < h['flag'])
    notdead = [h for h in hasgoodflag if 0 < h['t'] < 10000]
    for h in notdead:
        if 300 < h['t'] < 1000:
            k = 'H_1'
            h['as'].update(**{k: ion_spt[k](Hit.in_experimental_units(t=h['t'], x=h['x'], y=h['y'] - 1.5))
                           .to_experimental_units()})
        elif 1400 < h['t'] < 3000:
            k = 'C_1'
            h['as'].update(**{k: ion_spt[k](Hit.in_experimental_units(t=h['t'], x=h['x'], y=h['y']))
                           .to_experimental_units()})
        elif 4200 < h['t'] < 8000:
            if 4200 < h['t'] < 6500:
                k = 'Br_1'
                h['as'].update(**{k: ion_spt[k](Hit.in_experimental_units(t=h['t'], x=h['x'], y=h['y'] - 5))
                               .to_experimental_units()})
            if 5800 < h['t'] < 8000:
                k = 'I_1'
                h['as'].update(**{k: ion_spt[k](Hit.in_experimental_units(t=h['t'], x=h['x'] + 0.5, y=h['y'] - 8))
                               .to_experimental_units()})
    return notdead


@udf(SpkHits)
def analyse_ehits(hits: List[dict]) -> List[dict]:
    th = in_degree(-30)
    hasgoodflag = ({'t': h['t'] - -168.921,
                    'x': 1.64 * (cos(th) * h['x'] - sin(th) * h['y'] - -1.5818),
                    'y': 1.63 * (sin(th) * h['x'] + cos(th) * h['y'] - 0.51687),
                    'as': {},
                    } for h in hits if not 14 < h['flag'])
    notdead = [h for h in hasgoodflag if 0 < h['t'] < 60]
    for h in notdead:
        if 15 < h['t'] < 30:
            h['as'].update(**{'e': ele_spt(Hit.in_experimental_units(t=h['t'], x=h['x'], y=h['y']))
                           .to_experimental_units()})
    return notdead


# %% connect to spark master & read root files
with builder.getOrCreate() as spark:
    globbed = iglob(targetfiles)
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
                # .cache()
                )

    (analyzed
     .write
     .option("maxRecordsPerFile", 10000)  # less than 10 MB assuming a record of 1 KB,
     .parquet(save_as)
     )
