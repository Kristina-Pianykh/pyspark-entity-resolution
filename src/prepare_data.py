from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark import SparkConf
from cleaning import *
from utils import write_parquet
import argparse


conf = SparkConf()
conf.set("spark.master", "local[*]")
conf.set("spark.executor.memory", "4g")
conf.set("spark.driver.memory", "4g")
conf.set("spark.executor.cores", "10")
conf.set("spark.driver.maxResultSize", "2g")
conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
# conf.set("spark.driver.bindAddress", "localhost")
conf.set("spark.ui.port", "4052")
spark = SparkSession.builder.config(conf=conf).getOrCreate()


if __name__ == "__main__":
    separator = "\n\n"
    upper_year = 2004
    lower_year = 1995
    venues = ["SIGMOD", "VLDB"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw",
        type=str,
        help="path of the raw dataset",
    )
    parser.add_argument(
        "--dest",
        type=str,
        help="destination for the cleaned up dataset",
    )
    args = parser.parse_args()
    print(f"Raw dataset path: {args.raw}")
    print(f"Destination path: {args.dest}")

    df_cleaned: DataFrame = prepare_dataset(
        spark=spark,
        path=args.raw,
        separator=separator,
        schema=schema,
        upper_year=upper_year,
        lower_year=lower_year,
        venues=venues,
    )

    write_parquet(df_cleaned, args.dest, coal=False)

    spark.stop()
