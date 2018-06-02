from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, ArrayType, MapType, StringType


__all__ = ('SpkAnalyzedHit', 'SpkHit', 'SpkHits')


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
