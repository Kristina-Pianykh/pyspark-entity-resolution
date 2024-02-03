from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import (
    col,
    array_intersect,
    array_union,
    size,
    levenshtein,
    split,
)
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    StringType,
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


def jacard_similarity(title1, title2):
    # since we know that there're no NULL titles, we can skip the check
    tokens1 = split(title1, "\s+")
    tokens2 = split(title2, "\s+")
    intersec = array_intersect(tokens1, tokens2)
    union = array_union(tokens1, tokens2)
    return size(intersec) / size(union)


def compute_sim_and_match(df: DataFrame) -> DataFrame:

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

    similarity_df = (
        df.filter(same_venue)
        .withColumn("scores", levenshtein("dblp_authors", "acm_authors"))
        .filter((col("scores") >= 0) & (col("scores") < 10))
        .filter(cond1 | cond2 | cond3)
    )

    duplicates_df = similarity_df.withColumn(
        "title_jaccard_sim", jacard_similarity(col("dblp_title"), col("acm_title"))
    ).filter((col("title_jaccard_sim") >= 0.6))

    return duplicates_df
