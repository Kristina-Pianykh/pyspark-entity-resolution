#!/usr/bin/env python
from pyspark.sql import SparkSession
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
from pyspark.sql.column import Column
import unicodedata
from typing import Optional
from pyspark.sql.functions import (
    split,
    trim,
    col,
    lower,
    udf,
    monotonically_increasing_id,
    regexp_replace,
)


nltk.download("stopwords")
stop_words = stopwords.words("english")


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


def load_df(spark: SparkSession, file_name, separator, schema) -> DataFrame:
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
    spark: SparkSession,
    path: str,
    separator: str,
    schema: StructType,
    upper_year: int,
    lower_year: int,
    venues: list[str],
    # dest: str,
) -> DataFrame:
    df = load_df(spark, file_name=path, separator=separator, schema=schema)
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
