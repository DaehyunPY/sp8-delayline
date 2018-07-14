from typing import Callable, Optional, NewType, NamedTuple, Mapping
from warnings import warn

from .units import to_electron_volt, in_milli_meter, in_nano_sec

try:
    import pyspark as _
except ImportError:
    pyspark_exists = False
    warn('Package PySpark is not exists!')
else:
    pyspark_exists = True


__all__ = ('Hit', 'AnalyzedHit', 'SpkAnalyzedHit', 'SpkHit', 'SpkHits')


class Hit(NamedTuple):
    """
    Defines a set of detected time, position x and y
    """
    t: float
    x: float
    y: float

    @staticmethod
    def in_experimental_units(t: float, x: float, y: float) -> 'Hit':
        """
        Initialize `Hit` in the experimental units: 'ns' for `t`, and 'mm' for `x` and `y`
        :param t: flight time
        :param x: detected x position
        :param y: detected y position
        :return: `Hit`
        """
        return Hit(t=in_nano_sec(t), x=in_milli_meter(x), y=in_milli_meter(y))


class AnalyzedHit(NamedTuple):
    """
    Defines a set of kinetic energy and momentum which are analyzed by a Model
    """
    px: float
    py: float
    pz: float
    ke: float

    def to_experimental_units(self) -> dict:
        """
        Return momentum and kinetic energy converted to the experimental units: 'au' for momentums `px`, `py` and `pz`,
        and 'eV' for kinetic energy `ke`
        :return: `dict`
        """
        return {'px': self.px, 'py': self.py, 'pz': self.pz, 'ke': to_electron_volt(self.ke)}


Model = NewType('Model', Callable[[Hit], Optional[AnalyzedHit]])
Analyzer = NewType('Analyzer', Callable[[Hit], Mapping[str, Optional[AnalyzedHit]]])


if pyspark_exists:
    from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, ArrayType, MapType, StringType
    SpkAnalyzedHit = StructType([
        StructField('px', DoubleType(), nullable=False),
        StructField('py', DoubleType(), nullable=False),
        StructField('pz', DoubleType(), nullable=False),
        StructField('ke', DoubleType(), nullable=False),
    ])
    SpkHit = StructType([
        StructField('t', DoubleType(), nullable=False),
        StructField('x', DoubleType(), nullable=False),
        StructField('y', DoubleType(), nullable=False),
        StructField('flag', IntegerType(), nullable=True),
        StructField('as', MapType(StringType(), SpkAnalyzedHit), nullable=True),
    ])
    SpkHits = ArrayType(SpkHit)
else:
    SpkAnalyzedHit, SpkHit, SpkHits = None, None, None
