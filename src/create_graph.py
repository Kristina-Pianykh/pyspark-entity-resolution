#!/usr/bin/env python
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import (
    col,
    first,
)
from pyspark.conf import SparkConf
import argparse
from utils import *
from similarity import *
from clustering import *


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
        "--duplicates_path",
        type=str,
    )
    parser.add_argument(
        "--raw_dblp",
        type=str,
    )
    parser.add_argument(
        "--raw_acm",
        type=str,
    )
    parser.add_argument(
        "--dest",
        type=str,
    )

    args = parser.parse_args()
    print(f"cleaned dblp dataset path: {args.raw_dblp}")
    print(f"cleaned acm dataset path: {args.raw_acm}")
    print(f"duplicate pairs path: {args.duplicates_path}")
    print(f"destination path: {args.dest}")

    duplicates_df_complete = spark.read.parquet(args.duplicates_path)
    dblp_raw_cleaned: DataFrame = spark.read.parquet(args.raw_dblp)
    acm_raw_cleaned: DataFrame = spark.read.parquet(args.raw_acm)

    clusters: list[list[tuple]] = resolve_entities(duplicates_df_complete)
    labeled_clusters: list[tuple] = flatten_clusters(assign_cluster_id(clusters))

    clustered_df = spark.createDataFrame(
        labeled_clusters, schema=["id", "title", "authors", "df_name", "cluster_id"]
    )

    cluster_repr = clustered_df.groupBy("cluster_id", "df_name").agg(
        first("id").alias("id")
    )

    raw_dblp = dblp_raw_cleaned.select("id", "value")
    raw_acm = acm_raw_cleaned.select("id", "value")

    joined_dblp = cluster_repr.filter(col("df_name") == "dblp").join(
        raw_dblp, on=["id"], how="left"
    )
    joined_acm = cluster_repr.filter(col("df_name") == "acm").join(
        raw_acm, on=["id"], how="left"
    )

    joined_df = joined_dblp.union(joined_acm)
    assert joined_dblp.count() + joined_acm.count() == joined_df.count()

    pivoted_df = (
        joined_df.groupBy("cluster_id")
        .pivot("df_name")
        .agg(
            first("id"),
            first("value"),
        )
        .drop("cluster_id", "dblp_first(id)", "acm_first(id)")
    )
    write_parquet(pivoted_df, args.dest, coal=False)
    write_csv(pivoted_df, args.dest, sep="\t", coal=True)

    spark.stop()
