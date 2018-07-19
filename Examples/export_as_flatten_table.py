# %% external dependencies
from ROOT import TFile, TTree
from array import array
from itertools import islice
from pyspark.sql import SparkSession

# %% initialize spark builder
maxhits = 8
builder = (SparkSession
           .builder
           .config("spark.jars.packages", "org.diana-hep:spark-root_2.11:0.1.15")
           # .config("spark.cores.max", 11)
           # .config("spark.executor.cores", 5)
           # .config("spark.executor.memory", "4g")
           )


# %%
with builder.getOrCreate() as spark:
    df = spark.read.parquet('test.parquet')
    f = TFile('test.root', 'NEW')
    tree = TTree('Events', 'Events')
    tag = array('i', [0])
    nhits = array('i', [0])
    tarr = [array('f', [0]) for _ in range(maxhits)]
    xarr = [array('f', [0]) for _ in range(maxhits)]
    yarr = [array('f', [0]) for _ in range(maxhits)]
    flagarr = [array('i', [0]) for _ in range(maxhits)]
    tree.Branch('Tag', tag, 'Tag/I')
    tree.Branch('IonNum', nhits, 'IonNum/I')
    for i in range(maxhits):
        tree.Branch(f'IonT{i}', tarr[i], f'IonT{i}/F')
        tree.Branch(f'IonX{i}', xarr[i], f'IonX{i}/F')
        tree.Branch(f'IonY{i}', yarr[i], f'IonY{i}/F')
        tree.Branch(f'IonFlag{i}', flagarr[i], f'IonFlag{i}/I')
    for d in df.toLocalIterator():
        tag[0] = d.tag
        nhits[0] = len(d.hits)
        for i, h in enumerate(islice(d.hits, maxhits)):
            tarr[i][0] = h.t
            xarr[i][0] = h.x
            yarr[i][0] = h.y
            flagarr[i][0] = h.flag
        tree.Fill()
    f.Write()
    f.Close()
