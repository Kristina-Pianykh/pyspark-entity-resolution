from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import lit
from collections import defaultdict


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
