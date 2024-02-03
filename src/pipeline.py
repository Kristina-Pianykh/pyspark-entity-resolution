#!/usr/bin/env python
import os
from collections import defaultdict
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    StringType,
)
import re
from string import punctuation
from nltk.corpus import stopwords
import nltk
import argparse
from pyspark.sql.column import Column
import unicodedata
from typing import Optional
from pyspark.sql.functions import (
    first,
    split,
    lit,
    col,
    array_intersect,
    array_union,
    size,
    levenshtein,
    trim,
    col,
    lower,
    udf,
    monotonically_increasing_id,
    regexp_replace,
)


nltk.download("stopwords")
stop_words = stopwords.words("english")


# # Create a SparkConf and set the necessary configurations
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

# spark = SparkSession.builder.getOrCreate()


schema = StructType(
    [
        StructField("raw", StringType(), True),
        StructField("title", StringType(), True),
        StructField("authors", StringType(), True),
        StructField("year", IntegerType(), True),
        StructField("publication venue", StringType(), True),
        StructField("index", StringType(), True),
        StructField("references", StringType(), True),
        # StructField("abstract", StringType(), True), # ignore abstract for now cause dblp filtered has none anyway
    ]
)

similarity_schema = StructType(
    [
        StructField("dblp_title", IntegerType(), True),
        StructField("dblp_authors", StringType(), True),
        StructField("dblp_year", IntegerType(), True),
        StructField("dblp_venue", IntegerType(), True),
        StructField("dblp_id", StringType(), True),
        StructField("dblp_num_authors", StringType(), True),
        StructField("acm_title", IntegerType(), True),
        StructField("acm_authors", IntegerType(), True),
        StructField("acm_year", IntegerType(), True),
        StructField("acm_venue", IntegerType(), True),
        StructField("acm_id", IntegerType(), True),
        StructField("acm_num_authors", IntegerType(), True),
        StructField("scores", IntegerType(), True),
        StructField("title_jaccard_sim", IntegerType(), True),
    ]
)

YEAR_UPPER_BOUND = 2004
YEAR_LOWER_BOUND = 1995
YEAR_BLOCK_SIZE = 3


def jacard_similarity(title1, title2):
    # since we know that there're no NULL titles, we can skip the check
    tokens1 = split(title1, "\s+")
    tokens2 = split(title2, "\s+")
    intersec = array_intersect(tokens1, tokens2)
    union = array_union(tokens1, tokens2)
    return size(intersec) / size(union)


def load_and_rename_df(spark: SparkSession, path: str, dataset_name: str) -> DataFrame:
    # df = spark.read.options(header="true", inferSchema="true", delimiter="\t").csv(path)
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


def compute_sim_and_match(df: DataFrame) -> DataFrame:
    # match records as referring to the same entity if:
    # 1. the similarity score is 0 but both records have at least one author
    # 2. the similarity score is between 1 and 2 and neither of the records is an empty string
    # 3. the similarity score is between 3 and 10 but only if both records have more than one author?
    # 4. when both authors are empty strings: resort to other attributes

    # some predefined conditions
    non_zero_authors = (col("dblp_num_authors") > 0) & (col("acm_num_authors") > 0)
    zero_authors = (col("dblp_num_authors") == 0) & (col("acm_num_authors") == 0)
    # more_than_one_author = (F.col("dblp_num_authors") > 1) & (F.col("acm_num_authors") > 1)
    same_venue = (
        col("dblp_venue").contains("sigmod") & col("acm_venue").contains("sigmod")
    ) | ((col("dblp_venue").contains("vldb")) & (col("acm_venue").contains("vldb")))
    same_num_authors = col("dblp_num_authors") == col("acm_num_authors")

    # matching conditions
    cond1 = (col("scores") == 0) & same_num_authors & non_zero_authors
    cond2 = (col("scores") == 0) & zero_authors
    cond3 = (col("scores") > 0) & (col("scores") < 10) & same_num_authors
    # cond4 = (col("scores") >= 3) & (col("scores") < 10) & same_num_authors

    similarity_df = (
        df.filter(same_venue)
        .withColumn("scores", levenshtein("dblp_authors", "acm_authors"))
        .filter((col("scores") >= 0) & (col("scores") < 10))
        .filter(cond1 | cond2 | cond3)
    )

    # similarity_df = similarity_df.filter(cond1 | cond2 | cond3)

    duplicates_df = similarity_df.withColumn(
        "title_jaccard_sim", jacard_similarity(col("dblp_title"), col("acm_title"))
    ).filter((col("title_jaccard_sim") >= 0.6))

    return duplicates_df


def write_parquet(df: DataFrame, dest: str, coalesce: bool = True) -> None:
    if coalesce:
        df.coalesce(1).write.option("compression", "snappy").mode("overwrite").parquet(
            dest
        )
    else:
        df.write.option("compression", "snappy").mode("overwrite").parquet(dest)


def write_csv(df: DataFrame, dest: str, sep: str, coal: bool = True) -> None:
    script_dir = os.path.dirname(__file__)
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


@udf(useArrow=True)
def sort_authors(s: Optional[str]) -> Optional[str]:
    if not isinstance(s, str) and s is None:
        return None
    authors = []
    for name in s.split(","):
        tokens = [i for i in name.split(" ") if i]
        tokens = sorted([token.strip() for token in tokens])
        name_str = " ".join(tokens)
        authors.append(name_str)
    return ", ".join(authors)


@udf(useArrow=True)
def remove_nums(s: Optional[str]) -> Optional[str]:
    if not isinstance(s, str) and s is None:
        return None
    return re.sub(r"\d", "", s)


@udf(useArrow=True)  # An Arrow Python UDF
def clean_record(s: Optional[str]) -> Optional[str]:
    if not isinstance(s, str) and s is None:
        return None
    stop_symbols = r'!"$%&\'()’:;+.<=>?″`“”/-@[\\]#{|}'  # preserve commas
    res = re.sub(f"[{re.escape(stop_symbols)}]", " ", s)
    new = re.sub(r"\s{2,}", " ", res)  # remove multiple spaces
    return new.strip()


@udf(StringType())
def remove_accents(record: Optional[str]):
    if not isinstance(record, str) and record is None:
        return None
    normalized_record = unicodedata.normalize("NFKD", record)
    return "".join(
        [char for char in normalized_record if not unicodedata.combining(char)]
    )


@udf(useArrow=True)
def remove_stopwords(s: Optional[str]) -> Optional[str]:
    quotes = "\"“'’″`“”"
    if not isinstance(s, str) and s is None:
        return None
    tokens = [
        word.replace('"', "").strip(punctuation + quotes + " ")
        for word in re.split(" |—", s)
        if word not in stop_words
    ]
    tokens_str = " ".join(tokens)
    no_multiple_spaces = re.sub(r"\s{2,}", " ", tokens_str)  # remove multiple spaces
    return no_multiple_spaces


def exract_column(text: DataFrame, unic_prefix: str) -> Column:
    if unic_prefix == "\n#%":
        items = split(text.raw, unic_prefix, 2).getItem(1)
        # text.withColumn("items", items).show(truncate=False)
        items = regexp_replace(items, "\n#%", ", ")
    else:
        items = split(text.raw, unic_prefix).getItem(1)
    return trim(split(items, "\n").getItem(0))


@udf(useArrow=True)
def get_num_authors(s: Optional[str]) -> Optional[int]:
    if not isinstance(s, str) and s is None:
        return 0
    return len(s.split(","))


def clean_df(spark: SparkSession, df: DataFrame) -> DataFrame:
    res = spark.createDataFrame(data=[], schema=schema)
    res = (
        df.withColumn("title", remove_stopwords(remove_accents(df.title)))
        .withColumn(
            "authors",
            sort_authors(remove_accents(clean_record(remove_nums(df.authors)))),
        )
        .withColumn("year", df.year)
        .withColumn("publication venue", clean_record(col("publication venue")))
        .withColumn("index", clean_record(df.index))
        .withColumn("references", clean_record(df.references))
        # .withColumn("abstract", clean_record(df.abstract))
    )
    res = res.withColumn("value", trim(regexp_replace("value", "\n", " ")))

    return res


def load_df(file_name, separator, schema) -> DataFrame:
    text = spark.read.text(file_name, lineSep=separator)
    text = text.withColumn("raw", lower(trim(text["value"])))

    title = exract_column(text, "#\*")
    authors = exract_column(text, "\n#@")
    year = exract_column(text, "\n#t")
    publication_venue = exract_column(text, "\n#c")
    index = exract_column(text, "\n#index")
    references = exract_column(text, "\n#%")
    # abstract = exract_column(text, "\n#\!")

    df = spark.createDataFrame(data=[], schema=schema)
    df = (
        text.withColumn("title", trim(title))
        .withColumn("authors", trim(authors))
        .withColumn("year", trim(year).cast("Integer"))
        .withColumn("publication venue", trim(publication_venue))
        .withColumn("index", trim(index))
        .withColumn("references", trim(references))
        # .withColumn("abstract", trim(abstract))
    )
    df = df.drop(df.raw)
    return df


def filter_by_year_and_venue(
    df: DataFrame, upper_year: int, lower_year: int, venues: list[str]
) -> DataFrame:
    venues_lower: list[str] = [venue.lower() for venue in venues]
    year_range = (col("year") >= lower_year) & (col("year") <= upper_year)
    venue_range = (col("publication venue").contains(venues_lower[0])) | (
        col("publication venue").contains(venues_lower[1])
    )
    filtered_by_year_and_venue = df.filter(year_range & venue_range)
    return filtered_by_year_and_venue


def prepare_dataset(
    path: str,
    separator: str,
    schema: StructType,
    upper_year: int,
    lower_year: int,
    venues: list[str],
    # dest: str,
) -> DataFrame:
    df = load_df(file_name=path, separator=separator, schema=schema)
    filtered_df = filter_by_year_and_venue(
        df=df, upper_year=upper_year, lower_year=lower_year, venues=venues
    )
    filtered_df = filtered_df.withColumn(
        "id", monotonically_increasing_id()
    )  # artificial id
    cleaned_df = clean_df(spark, filtered_df).withColumns(
        {
            "id": monotonically_increasing_id(),
            "num_authors": get_num_authors(col("authors")),
        }
    )
    # df = df.drop(df.abstract)  # ignore abstract for now cause dblp filtered has none anyway
    return cleaned_df


def resolve_entities(df: DataFrame) -> list[list[tuple]]:
    visited = set()
    cluster = []
    clusters: list[list[tuple]] = []
    adj_matrix = defaultdict(list)

    df1 = df.withColumns({"dblp": lit("dblp"), "acm": lit("acm")}).collect()

    def create_matrix(
        idx1, idx2, title1, title2, authors1, authors2, df_name1, df_name2
    ):
        adj_matrix[(idx1, title1, authors1, df_name1)].append(
            (idx2, title2, authors2, df_name2)
        )
        adj_matrix[(idx2, title2, authors2, df_name2)].append(
            (idx1, title1, authors1, df_name1)
        )

    def dfs(node: tuple[int, str, str]):
        if (node) in visited:
            return
        visited.add(node)
        cluster.append(node)
        for neighbor in adj_matrix[node]:
            dfs(neighbor)

    for row in df1:
        create_matrix(
            row["dblp_id"],
            row["acm_id"],
            row["dblp_authors"],
            row["acm_authors"],
            row["dblp_title"],
            row["acm_title"],
            row["dblp"],
            row["acm"],
        )

    for val in adj_matrix.values():
        assert len(val) >= 1

    for node in adj_matrix.keys():
        dfs(node)
        if cluster:
            assert len(cluster) > 1
            clusters.append(cluster)
        cluster = []

    return clusters


def assign_cluster_id(clusters: list[list[tuple]]) -> list[list[tuple]]:
    labeled_clusters: list[list[tuple]] = []
    for cluster_enum, cluster in enumerate(clusters):
        labeled_cluster: list[tuple] = []
        for node in cluster:
            node = list(node)
            node.append(cluster_enum)
            node = tuple(node)
            labeled_cluster.append(node)
        labeled_clusters.append(labeled_cluster)

    return labeled_clusters


def flatten_clusters(clusters: list[list[tuple]]) -> list[tuple]:
    flattened_clusters: list[tuple] = []
    for cluster in clusters:
        flattened_clusters.extend(cluster)
    return flattened_clusters


def measure_performance(
    df_complete: DataFrame, df_blocked: DataFrame
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    df_complete = df_complete.select("dblp_id", "acm_id")
    df_blocked = df_blocked.select("dblp_id", "acm_id")

    metrics["true duplicates"] = df_complete.count()
    metrics["blocked duplicates"] = df_blocked.count()
    metrics["true positives"] = df_complete.intersect(df_blocked).count()
    metrics["false negatives"] = df_complete.subtract(df_blocked).count()
    metrics["false positives"] = df_blocked.subtract(df_complete).count()
    metrics["precision"] = metrics["true positives"] / (
        metrics["true positives"] + metrics["false positives"]
    )
    metrics["recall"] = metrics["true positives"] / (
        metrics["true positives"] + metrics["false negatives"]
    )
    metrics["f1"] = (
        2
        * (metrics["precision"] * metrics["recall"])
        / (metrics["precision"] + metrics["recall"])
    )

    metrics["precision"] = round(metrics["precision"], 3)
    metrics["recall"] = round(metrics["recall"], 3)
    metrics["f1"] = round(metrics["f1"], 2)

    return metrics


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--year_range",
        type=int,
        help="flag to indicate whether to run matching on all pairs without blocking",
    )
    args = args.parse_args()
    # metrics_path = "data/matched_entities/metrics/"

    if not args.year_range:
        result_path = "data/matched_entities/full/"
    else:
        result_path = "data/matched_entities/blocked/"
        if args.year_range > 9:
            YEAR_BLOCK_SIZE = 9
            print(f"YEAR_BLOCK_SIZE: {YEAR_BLOCK_SIZE}")
        elif args.year_range <= 0:
            print("YEAR_BLOCK_SIZE must be positive")
            exit(1)
        else:
            YEAR_BLOCK_SIZE = args.year_range
            print(f"YEAR_BLOCK_SIZE: {YEAR_BLOCK_SIZE}")

    print(
        f"+++++++++++++++++++++++++++result_path: {result_path}+++++++++++++++++++++++++++"
    )
    print("spark conf:")
    print(spark.sparkContext.getConf().getAll())
    separator = "\n\n"
    upper_year = 2004
    lower_year = 1995
    venues = ["sigmod", "vldb"]
    clusters_by_year_and_venue: list[list[tuple]] = []
    duplicates_df_complete = spark.createDataFrame([], schema=similarity_schema).cache()

    dblp_path = "data/dblp.txt"
    acm_path = "data/citation-acm-v8.txt"

    dblp_raw_cleaned: DataFrame = prepare_dataset(
        path=dblp_path,
        separator=separator,
        schema=schema,
        upper_year=upper_year,
        lower_year=lower_year,
        venues=venues,
    ).cache()
    # # dblp_raw_cleaned.show(vertical=True, truncate=False)
    # write_parquet(dblp_raw_cleaned, "data/dblp_cleaned_tmp")

    acm_raw_cleaned: DataFrame = prepare_dataset(
        path=acm_path,
        separator=separator,
        schema=schema,
        upper_year=upper_year,
        lower_year=lower_year,
        venues=venues,
    ).cache()

    dblp_df = rename_columns(
        dblp_raw_cleaned.drop("value", "index", "references"), "dblp"
    ).cache()
    acm_df = rename_columns(
        acm_raw_cleaned.drop("value", "index", "references"), "acm"
    ).cache()

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

                dblp_block = dblp_df.filter(filter_cond).cache()
                acm_block = acm_df.filter(filter_cond).cache()

                df = dblp_block.crossJoin(acm_block).cache()

                duplicates_df = compute_sim_and_match(df).cache()
                duplicates_df_complete = (
                    duplicates_df_complete.union(duplicates_df).distinct().cache()
                )

    duplicates_df_complete = duplicates_df_complete.select(
        "dblp_id", "acm_id", "dblp_title", "acm_title", "dblp_authors", "acm_authors"
    ).cache()

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
    write_csv(pivoted_df, result_path, "\t", coal=True)

    spark.stop()
