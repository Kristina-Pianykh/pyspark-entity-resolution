# Entity Resolution with PySpark

A data pipeline that resolves the records in two publication datasets that refer to the same entity. Under the hood, it relies on Levenshtein edit distance and Jaccard similarity coefficient for fuzzy string matching.

## How to Run 

Launch the cluster:

```bash
docker-compose up
```

Note the IP address of the master node assigned in the Docker network.
Submit a task to the master node:

```bash
docker-compose exec spark-master spark-submit \
      --master spark://<master_node_ip>:7077 \
      --driver-memory 4000M  \
      --executor-memory 3000M \
      --executor-cores 1 \
        src/pipeline.py [--year_range <range>]
```

Omitting `--year_range` will run the entity resolution algorithm on all the possible pairs across the datasets. Passing an `int`to `--year_range` between `1` and `9` will run the algortithm on batches of data with the time window of `<range>` years.
