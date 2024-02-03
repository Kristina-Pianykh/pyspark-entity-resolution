from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark import SparkConf
from pyspark.sql.functions import col
from similarity import *
from utils import write_parquet, rename_columns
import argparse


conf = SparkConf()
conf.set("spark.master", "local[*]")
conf.set("spark.executor.memory", "4g")
conf.set("spark.driver.memory", "4g")
conf.set("spark.driver.maxResultSize", "2g")
conf.set("spark.executor.cores", "10")
conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
# conf.set("spark.driver.bindAddress", "localhost")
conf.set("spark.ui.port", "4052")
spark = SparkSession.builder.config(conf=conf).getOrCreate()


YEAR_UPPER_BOUND = 2004
YEAR_LOWER_BOUND = 1995
YEAR_BLOCK_SIZE = 3


def process_venue_year(venue_year):
    venue, start_year = venue_year
    end_year = start_year + YEAR_BLOCK_SIZE
    print(f"venue: {venue}, year range: {start_year}-{end_year}")

    # Filter conditions
    same_venue = col("publication venue").contains(venue)
    year_range = (col("year") >= start_year) & (col("year") <= end_year)
    filter_cond = year_range & same_venue

    dblp_block = dblp_df.filter(filter_cond).cache()
    acm_block = acm_df.filter(filter_cond).cache()
    df = dblp_block.crossJoin(acm_block).cache()
    duplicates_df = compute_sim_and_match(df).cache()
    
    return duplicates_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dblp_path",
        type=str,
    )
    parser.add_argument(
        "--acm_path",
        type=str,
    )
    parser.add_argument(
        "--dest",
        type=str,
    )
    parser.add_argument(
        "--year_range",
        type=int,
        help="flag to indicate whether to run matching on all pairs without blocking",
    )

    args = parser.parse_args()
    print(f"cleaned dblp dataset path: {args.dblp_path}")
    print(f"cleaned acm dataset path: {args.acm_path}")
    print(f"destination path: {args.dest}")

    if args.year_range:
        if args.year_range > 9:
            YEAR_BLOCK_SIZE = 9
            print(f"YEAR_BLOCK_SIZE: {YEAR_BLOCK_SIZE}")
        elif args.year_range <= 0:
            print("YEAR_BLOCK_SIZE must be positive")
            exit(1)
        else:
            YEAR_BLOCK_SIZE = args.year_range
            print(f"YEAR_BLOCK_SIZE: {YEAR_BLOCK_SIZE}")

    venues = ["sigmod", "vldb"]
    clusters_by_year_and_venue: list[list[tuple]] = []
    duplicates_df_complete = spark.createDataFrame([], schema=similarity_schema)

    dblp_raw_cleaned: DataFrame = spark.read.parquet(args.dblp_path)
    acm_raw_cleaned: DataFrame = spark.read.parquet(args.acm_path)

    dblp_df = rename_columns(
        dblp_raw_cleaned.drop("value", "index", "references"), "dblp"
    )
    acm_df = rename_columns(acm_raw_cleaned.drop("value", "index", "references"), "acm")

    if not args.year_range:
        df = dblp_df.crossJoin(acm_df)
        duplicates_df_complete = compute_sim_and_match(df)
    else:
        # we block by the venue and a range of YEAR_BLOCK_SIZE years
        for venue in venues:
            print(f"venue: {venue}")
            same_venue = col("publication venue").contains(venue)

            for start_year in range(YEAR_LOWER_BOUND, YEAR_UPPER_BOUND):
                end_year = start_year + YEAR_BLOCK_SIZE

                if end_year > YEAR_UPPER_BOUND:
                    break

                print(f"year range: {start_year}-{end_year}")
                year_range = (col("year") >= start_year) & (col("year") <= end_year)
                filter_cond = year_range & same_venue

                dblp_block = dblp_df.filter(filter_cond)
                acm_block = acm_df.filter(filter_cond)

                df = dblp_block.crossJoin(acm_block)

                duplicates_df = compute_sim_and_match(df)
                duplicates_df_complete = duplicates_df_complete.union(
                    duplicates_df
                ).distinct()

    duplicates_df_complete = duplicates_df_complete.select(
        "dblp_id", "acm_id", "dblp_title", "acm_title", "dblp_authors", "acm_authors"
    )

    write_parquet(duplicates_df_complete, args.dest, coal=False)
    spark.stop()
