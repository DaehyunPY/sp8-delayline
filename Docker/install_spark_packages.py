#!/usr/bin/env python3
from pyspark.sql import SparkSession
builder = (SparkSession
           .builder
           .config("spark.jars.packages", "org.diana-hep:spark-root_2.11:0.1.15")
           .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
           )
with builder.getOrCreate():
    pass
