# %% external dependencies
from array import array

from ROOT import TFile, TTree
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType

# %% initialize spark builder
builder = (SparkSession
           .builder
           .config("spark.jars.packages", "org.diana-hep:spark-root_2.11:0.1.15")
           # .config("spark.cores.max", 11)
           # .config("spark.executor.cores", 5)
           # .config("spark.executor.memory", "4g")
           )
spark = builder.getOrCreate()

# %%
df = spark.read.parquet('test.parquet')
maxhits = 8


# %%
@udf(StructType([StructField('IonNum', IntegerType(), nullable=False),
                 *[StructField(f'IonT{i}', DoubleType(), nullable=True) for i in range(maxhits)],
                 *[StructField(f'IonX{i}', DoubleType(), nullable=True) for i in range(maxhits)],
                 *[StructField(f'IonY{i}', DoubleType(), nullable=True) for i in range(maxhits)],
                 *[StructField(f'IonFlag{i}', IntegerType(), nullable=True) for i in range(maxhits)]]))
def flat(hits):
    return {
        'IonNum': len(hits),
        **{f'IonT{i}': h.t for i, h in enumerate(hits)},
        **{f'IonX{i}': h.x for i, h in enumerate(hits)},
        **{f'IonY{i}': h.y for i, h in enumerate(hits)},
        **{f'IonFlag{i}': h.flag for i, h in enumerate(hits)},
    }


flatten = (
    df
        .withColumn('flatten', flat('hits'))
        .withColumnRenamed('tag', 'Tag')
        .select('Tag', 'flatten.*')
)

# %% to csv
(flatten
 .write
 .option("maxRecordsPerFile", 100000)
 .csv('test.csv', header=True, nullValue=0))

# %% to root
f = TFile('test.root', 'NEW')
tree = TTree('Events', 'Events')

tag = array('i', [0])
nhits = array('i', [0])
tarr = tuple(array('f', [0]) for _ in range(maxhits))
xarr = tuple(array('f', [0]) for _ in range(maxhits))
yarr = tuple(array('f', [0]) for _ in range(maxhits))
flagarr = tuple(array('i', [0]) for _ in range(maxhits))

tree.Branch('Tag', tag, 'Tag/I')
tree.Branch('IonNum', nhits, 'IonNum/I')
for i in range(maxhits):
    tree.Branch(f'IonT{i}', tarr[i], f'IonT{i}/F')
    tree.Branch(f'IonX{i}', xarr[i], f'IonX{i}/F')
    tree.Branch(f'IonY{i}', yarr[i], f'IonY{i}/F')
    tree.Branch(f'IonFlag{i}', flagarr[i], f'IonFlag{i}/I')

for d in flatten.toLocalIterator():
    tag[0] = d.Tag
    nhits[0] = d.IonNum
    for i in range(min(d.IonNum, maxhits)):
        tarr[i][0] = getattr(d, f'IonT{i}')
        xarr[i][0] = getattr(d, f'IonX{i}')
        yarr[i][0] = getattr(d, f'IonY{i}')
        flagarr[i][0] = getattr(d, f'IonFlag{i}')
    tree.Fill()
f.Write()
f.Close()
