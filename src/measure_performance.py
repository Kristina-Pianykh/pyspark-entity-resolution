from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.errors.exceptions.captured import AnalysisException
from similarity import *
from performance import measure_performance
import argparse


conf = SparkConf()
conf.set("spark.master", "local[*]")
conf.set("spark.executor.memory", "4g")
conf.set("spark.driver.memory", "4g")
conf.set("spark.executor.cores", "10")
conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
# conf.set("spark.driver.bindAddress", "localhost")
conf.set("spark.ui.port", "4052")
spark = SparkSession.builder.config(conf=conf).getOrCreate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--block_path",
        type=str,
    )
    parser.add_argument(
        "--full_path",
        type=str,
    )

    args = parser.parse_args()
    print(args.block_path)
    print(args.full_path)
    
    try:
        blocked_df = spark.read.parquet(args.block_path)
    except AnalysisException:
        print(f"File {args.block_path} does not exist. Make sure to run matching with blocking first.")
        exit(1)
    try:
        full_df = spark.read.parquet(args.full_path)
    except AnalysisException:
        print(f"File {args.full_path} does not exist. Make sure to run matching on the entire dataset first.")
        exit(1)

    metrics = measure_performance(df_complete=full_df, df_blocked=blocked_df)
    for metric, val in metrics.items():
        print(f"{metric}: {val}")

    spark.stop()
