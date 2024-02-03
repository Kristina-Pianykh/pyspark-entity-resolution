from pyspark.sql.dataframe import DataFrame


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
    metrics["f1"] = round(metrics["f1"], 3)

    return metrics
