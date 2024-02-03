import os
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col


def load_and_rename_df(spark: SparkSession, path: str, dataset_name: str) -> DataFrame:
    df = spark.read.parquet(path)
    df = df.select(
        [
            col("authors").alias(f"{dataset_name}_authors"),
            col("id").alias(f"{dataset_name}_id"),
            col("num_authors").alias(f"{dataset_name}_num_authors"),
            col("year").alias(f"{dataset_name}_year"),
            col("publication venue").alias(f"{dataset_name}_venue"),
            col("title").alias(f"{dataset_name}_title"),
            col("value").alias(f"{dataset_name}_value"),
            # col("references").alias(f"{dataset_name}_references"),
        ]
    )
    return df


def rename_columns(df: DataFrame, dataset_name: str) -> DataFrame:
    df = df.withColumnRenamed("publication venue", "venue")
    for column in df.columns:
        df = df.withColumnRenamed(column, f"{dataset_name}_{column}")
    return df


def write_parquet(df: DataFrame, dest: str, coal: bool = True) -> None:
    if coal:
        df.coalesce(1).write.option("compression", "snappy").mode("overwrite").parquet(
            dest
        )
    else:
        df.write.option("compression", "snappy").mode("overwrite").parquet(dest)


def write_csv(df: DataFrame, dest: str, sep: str, coal: bool = True) -> None:
    new_name = "Matched_Entities.csv"
    if coal:
        df.coalesce(1).write.options(header="true", delimiter=sep).mode(
            "overwrite"
        ).csv(dest)
    else:
        df.write.options(header="true", delimiter=sep).mode("overwrite").csv(dest)

    for file in os.listdir(dest):
        if file.endswith(".csv"):
            os.rename(
                os.path.join(dest, file), os.path.join(dest, new_name)
            )
            break
